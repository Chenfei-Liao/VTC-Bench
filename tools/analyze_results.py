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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOWNSAMPLE_TO_BUDGET = {
    1: '1', 2: '0.25', 3: '0.1111', 4: '0.0625', 5: '0.04', 10: '0.01'
}
BUDGET_TO_DOWNSAMPLE = {v: k for k, v in DOWNSAMPLE_TO_BUDGET.items()}

QUESTION_FIELDS = ['question', 'prompt', 'query', 'instruction', 'text']
CATEGORY_FIELDS = ['category', 'task', 'task_name', 'dataset']
L2_CATEGORY_FIELDS = ['l2_category', 'sub_category', 'subcategory', 'type', 'topic']


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
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
    Check if a response is correct based on dataset-specific metrics.
    """
    dataset_name_lower = dataset_name.lower()
    
    try:
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
        
        # Simple string comparison as fallback
        norm_response = str(response).replace('.', '').strip().lower()
        norm_target = str(target).replace('.', '').strip().lower()
        
        return norm_response == norm_target or norm_target in norm_response
        
    except Exception as e:
        logger.warning(f"Error evaluating correctness for dataset {dataset_name}: {e}")
        return False

def extract_field_value(doc: Dict[str, Any], field_options: List[str], default_value: str = 'unknown') -> str:
    for field in field_options:
        value = doc.get(field)
        if value is not None: return value
    return default_value

def get_question_text(doc: Dict[str, Any]) -> str:
    return extract_field_value(doc, QUESTION_FIELDS, '')

def get_category(doc: Dict[str, Any]) -> str:
    return extract_field_value(doc, CATEGORY_FIELDS, 'unknown')

def get_l2_category(doc: Dict[str, Any]) -> str:
    return extract_field_value(doc, L2_CATEGORY_FIELDS, 'unknown')

def get_clean_dataset_name(filename: str) -> str:
    name = filename.replace('.jsonl', '')
    if '_samples_' in name:
        return name.split('_samples_')[-1].lower()
    return name.lower()

def analyze_dataset_groups(origin_path: Path, downsample_path: Path, dataset_name: str) -> Tuple[List[str], List[str], Dict]:
    origin_data = load_jsonl(origin_path)
    down_data = load_jsonl(downsample_path)
    
    # Use string conversion for doc_id to match original script logic but prevent type mismatches
    origin_dict = {str(item['doc_id']): item for item in origin_data}
    down_dict = {str(item['doc_id']): item for item in down_data}
    
    group_b_ids, group_a_ids = [], []
    for doc_id, origin_item in origin_dict.items():
        if doc_id not in down_dict: continue
        
        if is_correct_by_dataset(origin_item, dataset_name):
            if is_correct_by_dataset(down_dict[doc_id], dataset_name):
                group_b_ids.append(doc_id)
            else:
                group_a_ids.append(doc_id)
                
    logger.info(f"Group A (Origin right & Downsample wrong): {len(group_a_ids)} samples")
    logger.info(f"Group B (Origin right & Downsample right): {len(group_b_ids)} samples")
    return group_a_ids, group_b_ids, origin_dict

def analyze_method_performance(method_path: Path, group_ids: List[str], origin_dict: Dict, dataset_name: str) -> Dict:
    method_data = load_jsonl(method_path)
    method_dict = {str(item['doc_id']): item for item in method_data}

    results = {'total': len(group_ids), 'correct': 0, 'incorrect': 0, 'accuracy': 0.0, 'details': []}

    for doc_id in group_ids:
        if doc_id not in method_dict: continue
            
        method_item = method_dict[doc_id]
        origin_item = origin_dict[doc_id]
        
        method_correct = is_correct_by_dataset(method_item, dataset_name)
        question_text = get_question_text(origin_item.get('doc', {}))
        truncated_question = question_text[:100] + '...' if len(question_text) > 100 else question_text
        
        detail_record = {
            'doc_id': doc_id,
            'question': truncated_question,
            'target': origin_item.get("doc", {}).get("answer", ""),
            'origin_response': origin_item.get("filtered_resps", ""),
            'method_response': method_item.get("filtered_resps", ""),
            'method_correct': method_correct,
            'category': get_category(origin_item.get('doc', {})),
            'l2_category': get_l2_category(origin_item.get('doc', {})),
        }
        
        results['details'].append(detail_record)
        if method_correct: results['correct'] += 1
        else: results['incorrect'] += 1

    if results['total'] > 0:
        results['accuracy'] = (results['correct'] / results['total']) * 100
    return results

def find_file_by_dataset(directory: Path, dataset_name: str) -> Optional[Path]:
    if not directory.exists(): return None
    for f in directory.glob("*.jsonl"):
        if get_clean_dataset_name(f.name) == dataset_name: return f
    return None

def main():
    parser = argparse.ArgumentParser(description='VTC-Bench Result Analysis')
    parser.add_argument('--base_dir', type=str, default='Data', help='Path to Data directory')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--all', action='store_true', help='Auto-scan all models and methods')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Directory to save outputs')
    parser.add_argument('--skip_details', action='store_true', help='Skip generating detailed CSV files')
    args = parser.parse_args()

    base_path, output_path = Path(args.base_dir), Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    models = [args.model] if args.model else [d.name for d in base_path.iterdir() if d.is_dir() and d.name != "analysis_results"]
    all_summaries = []

    for model in models:
        model_dir = base_path / model
        baseline_root = next((model_dir / n for n in ['Downsample', 'downsample'] if (model_dir / n).exists()), None)
        if not baseline_root: continue

        for method_dir in (d for d in model_dir.iterdir() if d.is_dir() and d.name.lower() != 'downsample'):
            method = method_dir.name
            for ratio_dir in (d for d in method_dir.iterdir() if d.is_dir()):
                budget = ratio_dir.name
                ratio = BUDGET_TO_DOWNSAMPLE.get(budget, budget)
                
                baseline_dir, origin_dir = baseline_root / str(ratio), baseline_root / '1'
                if not baseline_dir.exists() or not origin_dir.exists(): continue

                for method_file in ratio_dir.glob("*.jsonl"):
                    ds_name = get_clean_dataset_name(method_file.name)
                    base_f = find_file_by_dataset(baseline_dir, ds_name)
                    orig_f = find_file_by_dataset(origin_dir, ds_name)
                    
                    if not base_f or not orig_f: continue
                    
                    logger.info(f"Analyzing {model} | {method} | Budget {budget} | Dataset {ds_name}")
                    
                    ga_ids, gb_ids, orig_dict = analyze_dataset_groups(orig_f, base_f, ds_name)
                    res_a = analyze_method_performance(method_file, ga_ids, orig_dict, ds_name)
                    res_b = analyze_method_performance(method_file, gb_ids, orig_dict, ds_name)
                    
                    if not args.skip_details:
                        all_details = []
                        for d in res_a['details']:
                            d.update({'group': 'A', 'dataset': ds_name, 'model': model})
                            all_details.append(d)
                        for d in res_b['details']:
                            d.update({'group': 'B', 'dataset': ds_name, 'model': model})
                            all_details.append(d)
                        csv_path = output_path / f"{model}_{method}_budget{budget}_{ds_name}_details.csv"
                        pd.DataFrame(all_details).to_csv(csv_path, index=False, encoding='utf-8')

                    all_summaries.append({
                        'model': model, 'dataset': ds_name, 'method': method, 'budget': budget,
                        'downsample_level': ratio, 'group_a_total': res_a['total'],
                        'group_a_correct': res_a['correct'], 'group_a_accuracy': res_a['accuracy'],
                        'group_b_total': res_b['total'], 'group_b_correct': res_b['correct'],
                        'group_b_accuracy': res_b['accuracy']
                    })

    if all_summaries:
        df = pd.DataFrame(all_summaries)
        csv_name = f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path / csv_name, index=False)
        print("\nAnalysis Summary:\n" + "="*120)
        print(df[['model', 'dataset', 'method', 'downsample_level', 'group_a_accuracy', 'group_b_accuracy']].to_string(index=False))
    else: print("No results generated from analysis.")

if __name__ == "__main__":
    main()
