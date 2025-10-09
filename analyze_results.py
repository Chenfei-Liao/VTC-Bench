#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DART Method Analysis Script (Batch Version)
- Analyzes method performance on two groups:
  Group A: Origin correct & Downsample correct
  Group B: Origin correct & Downsample wrong
- Supports multiple methods and downsample levels
- Uses dataset-specific evaluation metrics
"""

import json
import re
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DOWNSAMPLE_RATIO_MAP = {2: '0.25', 3: '0.1111', 4: '0.0625', 5: '0.04', 10: '0.01'}
METHOD_DIRNAME_MAP = {
    'dart': 'dart',
    'fastv': 'fastv',
    'prumerge+': 'prumerge+',
    'visionzip': 'visionzip'
}
VALID_METHODS = list(METHOD_DIRNAME_MAP.keys())
VALID_DOWNSAMPLES = list(DOWNSAMPLE_RATIO_MAP.keys())

# Field name constants
QUESTION_FIELDS = ['question', 'prompt', 'query', 'instruction', 'text']
CATEGORY_FIELDS = ['category', 'task', 'task_name', 'dataset']
L2_CATEGORY_FIELDS = ['l2_category', 'sub_category', 'subcategory', 'type', 'topic']


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in {file_path} at line {line_num}: {e}")
        logger.info(f"Loaded {len(data)} records from {file_path}")
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise
    return data


def is_correct_by_dataset(doc: Dict[str, Any], dataset_name: str) -> bool:
    """
    Check if a response is correct based on dataset-specific evaluation metrics.
    
    Args:
        doc: The document containing response and evaluation data
        dataset_name: Name of the dataset to determine evaluation method
        
    Returns:
        bool: True if response is correct, False otherwise
    """
    dataset_name_lower = dataset_name.lower()
    
    try:
        # ChartQA evaluation
        if 'chartqa' in dataset_name_lower:
            relaxed_overall = doc.get('relaxed_overall')
            if relaxed_overall is not None:
                return bool(relaxed_overall)
        
        # POPE evaluation
        elif 'pope' in dataset_name_lower:
            average = doc.get('pope_accuracy', {})
            score = average.get('score')
            if score is not None:
                return score >= 0.5  # Typically 1 for correct, 0 for incorrect
        
        # OCRBench evaluation
        elif 'ocrbench' in dataset_name_lower:
            ocrbench_accuracy = doc.get('ocrbench_accuracy', {})
            score = ocrbench_accuracy.get('score')
            if score is not None:
                return score >= 0.5
        
        # GQA evaluation
        elif 'gqa' in dataset_name_lower:
            exact_match = doc.get('exact_match')
          
            if exact_match is not None:
                return bool(exact_match)
        
        # MME evaluation
        elif 'mme' in dataset_name_lower:
            # Check both perception and cognition scores
            perception_score = doc.get('mme_perception_score', {}).get('score')
            cognition_score = doc.get('mme_cognition_score', {}).get('score')
            
            if perception_score is not None:
                return perception_score >= 0.5
            elif cognition_score is not None:
                return cognition_score >= 0.5
        
        # MMStar evaluation
        elif 'mmstar' in dataset_name_lower:
            average = doc.get('average', {})
            score = average.get('score')
            if score is not None:
                return score >= 0.5
        
        # MMBench evaluation
        elif 'mmbench' in dataset_name_lower:
            gpt_eval_score = doc.get('gpt_eval_score', {})
            answer = gpt_eval_score.get('answer', '')
            prediction = gpt_eval_score.get('prediction', '')
            
            if answer and prediction:
                # Normalize by removing periods and extra spaces, convert to lowercase
                norm_answer = answer.replace('.', '').strip().lower()
                norm_prediction = prediction.replace('.', '').strip().lower()
                return norm_answer == norm_prediction
        
        # Fallback: use traditional answer comparison if no dataset-specific metric found
        response = doc.get('filtered_resps', '')
        target = doc.get('doc', {}).get('answer', '')
        
        #if isinstance(response, list):
        #    response = response[0] if response else ""
        #if isinstance(target, list):
        #    target = target[0] if target else ""
        
        # Simple string comparison as fallback
        #norm_response = str(response).replace('.', '').strip().lower()
        #norm_target = str(target).replace('.', '').strip().lower()
        
        #return norm_response == norm_target or norm_target in norm_response
        
    except Exception as e:
        logger.warning(f"Error evaluating correctness for dataset {dataset_name}: {e}")
        return False


def extract_field_value(doc: Dict[str, Any], field_options: List[str], default_value: str = 'unknown') -> str:
    """Extract field value from document using multiple possible field names."""
    for field in field_options:
        value = doc.get(field)
        if value is not None:
            return value
    return default_value


def get_question_text(doc: Dict[str, Any]) -> str:
    """Extract question text from document."""
    return extract_field_value(doc, QUESTION_FIELDS, '')


def get_category(doc: Dict[str, Any]) -> str:
    """Extract category from document."""
    return extract_field_value(doc, CATEGORY_FIELDS, 'unknown')


def get_l2_category(doc: Dict[str, Any]) -> str:
    """Extract level 2 category from document."""
    return extract_field_value(doc, L2_CATEGORY_FIELDS, 'unknown')


def discover_datasets(base_dir: Path, model_name: str, method: str, downsample: int) -> Dict[str, Dict[str, Path]]:
    """Discover dataset triplets in directory structure."""
    model_dir = base_dir / model_name
    

    method_ratio = DOWNSAMPLE_RATIO_MAP[downsample]
    method_dir = model_dir / method / method_ratio
    
    downsample_dir = model_dir / 'downsample' / str(downsample)
    
    origin_dir = None
    
    potential_origin = model_dir / 'origin'
    if potential_origin.is_dir():
        origin_dir = potential_origin
    else:
        origin_candidates = [
            model_dir / 'downsample' / '1', 
            model_dir / method / '1.0',      
        ]
        
        for candidate in origin_candidates:
            if candidate.is_dir():
                origin_dir = candidate
                logger.info(f"Using {origin_dir} as origin directory")
                break
        
   
        if origin_dir is None:
            for method_name in METHOD_DIRNAME_MAP.values():
                method_path = model_dir / method_name
                if method_path.is_dir():
                    
                    for ratio_dir in method_path.iterdir():
                        if ratio_dir.is_dir():
                            origin_dir = ratio_dir
                            logger.info(f"Using {origin_dir} as origin directory")
                            break
                    if origin_dir:
                        break

    missing_dirs = []
    for dir_path, dir_name in [
        (method_dir, f"{method}/{method_ratio}"),
        (downsample_dir, f"downsample/{downsample}"),
    ]:
        if not dir_path.is_dir():
            missing_dirs.append(f"{dir_name}: {dir_path}")
    
    if origin_dir is None:
        missing_dirs.append(f"origin: No suitable origin directory found")
    
    if missing_dirs:
        logger.warning(f"Missing directories: {', '.join(missing_dirs)}")
        return {}

    def index_directory_files(dir_path: Path) -> Dict[str, List[Path]]:
        """Index files in directory by dataset name."""
        index = defaultdict(list)
        if not dir_path.is_dir():
            return index
            
        for file_path in dir_path.glob("*.jsonl"):
            file_name = file_path.name.lower()
            if '_samples_' not in file_name:
                continue
            dataset_name = file_name[file_name.rfind('_samples_') + len('_samples_'):-len('.jsonl')]
            if not dataset_name:
                continue
            index[dataset_name].append(file_path)
        return index

    # Index files in each directory
    origin_index = index_directory_files(origin_dir)
    downsample_index = index_directory_files(downsample_dir)
    method_index = index_directory_files(method_dir)

    def select_latest_file(file_paths: List[Path]) -> Path:
        """Select the most recent file from a list of paths."""
        if not file_paths:
            raise ValueError("No files to select from")
        # choose the newest
        return sorted(file_paths)[-1]

    # Find common datasets
    common_datasets = set(origin_index.keys()) & set(downsample_index.keys()) & set(method_index.keys())
    dataset_map = {}
    
    for dataset_name in sorted(common_datasets):
        try:
            dataset_map[dataset_name] = {
                'origin': select_latest_file(origin_index[dataset_name]),
                'downsample': select_latest_file(downsample_index[dataset_name]),
                'method': select_latest_file(method_index[dataset_name]),
            }
        except (ValueError, IndexError) as e:
            logger.warning(f"Skipping dataset {dataset_name} due to file selection error: {e}")
            continue
            
    logger.info(f"Discovered {len(dataset_map)} datasets: {list(dataset_map.keys())}")
    return dataset_map


def analyze_dataset_groups(origin_path: Path, downsample_path: Path, dataset_name: str) -> Tuple[List[str], List[str]]:
    """Analyze origin and downsample to create groups A and B using dataset-specific metrics."""
    origin_data = load_jsonl(origin_path)
    downsample_data = load_jsonl(downsample_path)

    # Create dictionaries for easy lookup by doc_id
    origin_dict = {item['doc_id']: item for item in origin_data}
    downsample_dict = {item['doc_id']: item for item in downsample_data}

    # Create groups
    group_b_ids = []  # Origin correct & Downsample correct
    group_a_ids = []  # Origin correct & Downsample wrong

    for doc_id in origin_dict:
        if doc_id not in downsample_dict:
            continue
            
        origin_item = origin_dict[doc_id]
        downsample_item = downsample_dict[doc_id]
        
        origin_correct = is_correct_by_dataset(origin_item, dataset_name)
        downsample_correct = is_correct_by_dataset(downsample_item, dataset_name)
        
        if origin_correct:
            if downsample_correct:
                group_b_ids.append(doc_id)
            else:
                group_a_ids.append(doc_id)

    logger.info(f"Group A (Origin right & Downsample right): {len(group_a_ids)} samples")
    logger.info(f"Group B (Origin right & Downsample wrong): {len(group_b_ids)} samples")
    
    return group_a_ids, group_b_ids


def analyze_method_performance(method_path: Path, group_ids: List[str], origin_dict: Dict, dataset_name: str) -> Dict:
    """Analyze method performance on a specific group using dataset-specific metrics."""
    method_data = load_jsonl(method_path)
    method_dict = {item['doc_id']: item for item in method_data}

    results = {
        'total': len(group_ids),
        'correct': 0,
        'incorrect': 0,
        'accuracy': 0.0,
        'details': []
    }

    for doc_id in group_ids:
        if doc_id not in method_dict:
            continue
            
        method_item = method_dict[doc_id]
        origin_item = origin_dict[doc_id]
        
        method_correct = is_correct_by_dataset(method_item, dataset_name)
        question_text = get_question_text(origin_item.get('doc', {}))
        truncated_question = question_text[:100] + '...' if len(question_text) > 100 else question_text
        
        detail_record = {
            'doc_id': doc_id,
            'question': truncated_question,
            'target': origin_item["doc"]["answer"],
            'origin_response': origin_item["filtered_resps"],
            'method_response': method_item["filtered_resps"],
            'method_correct': method_correct,
            'category': get_category(origin_item.get('doc', {})),
            'l2_category': get_l2_category(origin_item.get('doc', {})),
        }
        
        results['details'].append(detail_record)
        if method_correct:
            results['correct'] += 1
        else:
            results['incorrect'] += 1

    if results['total'] > 0:
        results['accuracy'] = (results['correct'] / results['total']) * 100

    return results


def run_analysis_for_method(
    base_dir: Path,
    model_name: str,
    method: str,
    downsample: int,
    output_dir: Path,
    skip_details: bool = False
) -> List[Dict]:
    """Run analysis for a specific method and downsample level."""
    logger.info(f"Analyzing model '{model_name}', method '{method}' with downsample '{downsample}'")
    
    # Discover datasets
    dataset_map = discover_datasets(base_dir, model_name, method, downsample)
    
    if not dataset_map:
        logger.warning(f"No datasets found for model '{model_name}', method '{method}' and downsample '{downsample}'")
        return []
    
    summaries = []
    
    for dataset_name, dataset_paths in dataset_map.items():
        try:
            logger.info(f"Analyzing dataset: {dataset_name}")
            
            # Load origin data for reference
            origin_data = load_jsonl(dataset_paths['origin'])
            origin_dict = {item['doc_id']: item for item in origin_data}
            
            # Create groups A and B using dataset-specific metrics
            group_a_ids, group_b_ids = analyze_dataset_groups(
                dataset_paths['origin'], 
                dataset_paths['downsample'],
                dataset_name
            )
            
            # Analyze method performance on both groups using dataset-specific metrics
            group_a_results = analyze_method_performance(
                dataset_paths['method'], 
                group_a_ids, 
                origin_dict,
                dataset_name
            )
            
            group_b_results = analyze_method_performance(
                dataset_paths['method'], 
                group_b_ids, 
                origin_dict,
                dataset_name
            )
            
            # Save detailed results if requested
            if not skip_details:
                # Combine details
                all_details = []
                for detail in group_a_results['details']:
                    detail['group'] = 'A'
                    detail['dataset'] = dataset_name
                    detail['model'] = model_name
                    all_details.append(detail)
                for detail in group_b_results['details']:
                    detail['group'] = 'B'
                    detail['dataset'] = dataset_name
                    detail['model'] = model_name
                    all_details.append(detail)
                
                # Save CSV
                csv_path = output_dir / f"{model_name}_{method}_downsample{downsample}_{dataset_name}_details.csv"
                pd.DataFrame(all_details).to_csv(csv_path, index=False, encoding='utf-8')
                logger.info(f"Saved detailed results to {csv_path}")
            
            # Create summary
            summary = {
                'model': model_name,
                'dataset': dataset_name,
                'method': method,
                'downsample_level': downsample,
                'group_a_total': group_a_results['total'],
                'group_a_correct': group_a_results['correct'],
                'group_a_accuracy': group_a_results['accuracy'],
                'group_b_total': group_b_results['total'],
                'group_b_correct': group_b_results['correct'],
                'group_b_accuracy': group_b_results['accuracy'],
            }
            
            summaries.append(summary)
            logger.info(f"Dataset {dataset_name}: Group A accuracy = {group_a_results['accuracy']:.2f}%, Group B accuracy = {group_b_results['accuracy']:.2f}%")
            
        except Exception as e:
            logger.error(f"Error analyzing dataset '{dataset_name}': {e}")
            import traceback
            traceback.print_exc()
    
    return summaries


def main() -> None:
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze method performance on groups A and B')
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default=str(Path(__file__).parent), 
        help='Base directory containing model directories'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='Qwen2-VL-7B-Instruct', 
        choices=['Qwen2-VL-7B-Instruct', 'Llava-ov-7B'],
        help='Model name to analyze'
    )
    parser.add_argument(
        '--method', 
        type=str, 
        default='dart', 
        choices=VALID_METHODS, 
        help='Method name to analyze'
    )
    parser.add_argument(
        '--downsample', 
        type=int, 
        default=2, 
        choices=VALID_DOWNSAMPLES, 
        help='Downsample level to analyze'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='', 
        help='Directory to save all outputs'
    )
    parser.add_argument(
        '--skip_details', 
        action='store_true', 
        help='Skip generating detailed CSV files'
    )
    parser.add_argument(
        '--batch_mode', 
        action='store_true', 
        help='Run analysis for all method and downsample combinations'
    )
    parser.add_argument(
        '--methods', 
        type=str, 
        nargs='+', 
        default=VALID_METHODS, 
        help='Methods to analyze in batch mode'
    )
    parser.add_argument(
        '--downsamples', 
        type=int, 
        nargs='+', 
        default=VALID_DOWNSAMPLES, 
        help='Downsample levels to analyze in batch mode'
    )
    parser.add_argument(
        '--models', 
        type=str, 
        nargs='+', 
        default=['Qwen2-VL-7B-Instruct', 'Llava-ov-7B'],
        help='Models to analyze in batch mode'
    )
    
    args = parser.parse_args()
    
    try:
        base_dir = Path(args.base_dir)
        
        # Set up output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = base_dir / f"analysis_results_{timestamp}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        all_summaries = []
        
        if args.batch_mode:
            # Batch analysis for multiple models, methods and downsamples
            for model in args.models:
                for method in args.methods:
                    for downsample in args.downsamples:
                        method_output_dir = output_dir / f"{model}_{method}_downsample{downsample}"
                        method_output_dir.mkdir(exist_ok=True)
                        
                        summaries = run_analysis_for_method(
                            base_dir, model, method, downsample, method_output_dir, args.skip_details
                        )
                        all_summaries.extend(summaries)
        else:
            # Single analysis
            summaries = run_analysis_for_method(
                base_dir, args.model, args.method, args.downsample, output_dir, args.skip_details
            )
            all_summaries.extend(summaries)
        
        # Save combined summary
        if all_summaries:
            summary_csv = output_dir / 'analysis_summary.csv'
            pd.DataFrame(all_summaries).to_csv(summary_csv, index=False, encoding='utf-8')
            logger.info(f"Saved analysis summary to {summary_csv}")
            
            # Print summary table
            print("\nAnalysis Summary:")
            print("=" * 120)
            df = pd.DataFrame(all_summaries)
            print(df[['model', 'dataset', 'method', 'downsample_level', 
                     'group_a_accuracy', 'group_b_accuracy']].to_string(index=False))
        else:
            logger.warning("No results generated from analysis")
                
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.exception("Detailed error traceback:")


if __name__ == "__main__":
    main()
