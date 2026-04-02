import os
import shutil
import re
from pathlib import Path

def reorganize_results():
    """
    Scans all models in logs/ and syncs results to Data/ directory.
    """
    log_root = Path("logs")
    data_root = Path("Data")

    if not log_root.exists():
        print(f"Source root not found: {log_root}")
        return

    # Scan each model directory in logs/
    for model_dir in log_root.iterdir():
        if not model_dir.is_dir() or model_dir.name == "analysis_results":
            continue
        
        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            dir_name = exp_dir.name
            method = ""
            value_key = ""

            # Parse experiment directory names (e.g., fastv_IR1_RR0.75 or NATIVE_IR2)
            rr_match = re.search(r"(.+)_IR\d+_RR([\d\.]+)", dir_name, re.IGNORECASE)
            ir_only_match = re.search(r"(.+)_IR(\d+)$", dir_name, re.IGNORECASE)

            if rr_match:
                raw_method = rr_match.group(1)
                if raw_method.lower() == "fastv":
                    method = "FastV"
                elif raw_method.lower() == "dart":
                    method = "DART"
                else:
                    method = raw_method
                
                # Convert reduction ratio to budget
                try:
                    reduction_ratio = float(rr_match.group(2))
                    budget = 1.0 - reduction_ratio
                    value_key = f"{budget:g}"
                except ValueError:
                    continue
                
            elif ir_only_match:
                raw_method = ir_only_match.group(1)
                if raw_method.upper() == "NATIVE":
                    method = "Downsample"
                else:
                    method = raw_method
                value_key = ir_only_match.group(2)
            
            else:
                continue

            # Locate and copy jsonl sample files
            for jsonl_file in exp_dir.rglob("*.jsonl"):
                name_match = re.search(r"samples_(.+)\.jsonl", jsonl_file.name)
                if not name_match:
                    continue
                
                dataset_name = name_match.group(1)
                dest_dir = data_root / model_name / method / value_key
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                target_path = dest_dir / f"{dataset_name}.jsonl"
                shutil.copy2(jsonl_file, target_path)
                print(f"  [Mapped] {method}/{value_key}/{dataset_name}.jsonl")

    # Copy Qwen2-VL-7B-Instruct Downsample data to Llava-ov-7B
    qwen_downsample_dir = data_root / "Qwen2-VL-7B-Instruct" / "Downsample"
    llava_downsample_dir = data_root / "Llava-ov-7B" / "Downsample"
    
    if qwen_downsample_dir.exists():
        print("\nCopying Qwen2-VL-7B-Instruct Downsample data to Llava-ov-7B...")
        if llava_downsample_dir.exists():
            shutil.rmtree(llava_downsample_dir)
        shutil.copytree(qwen_downsample_dir, llava_downsample_dir)
        print("  [Copied] Qwen2-VL-7B-Instruct/Downsample -> Llava-ov-7B/Downsample")

if __name__ == "__main__":
    reorganize_results()
