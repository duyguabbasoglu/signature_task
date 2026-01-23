
import cv2
import numpy as np
import importlib.util
from pathlib import Path
from enum import Enum

# Define Ground Truth Categories
class Truth(Enum):
    EMPTY = "EMPTY"
    PUNCT = "PUNCT"
    SIGN = "SIGN"

def load_module(file_path):
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_ground_truth(filename):
    name = filename.lower()
    if "empty" in name:
        return Truth.EMPTY
    
    # Punctuation files
    punct_names = [
        "checkmark.png", "comma.png", "dash.png", "dot_1.png", 
        "dot_2.png", "dot_large.png", "dot_small.png", 
        "exclamation.png", "plus.png", "x_mark.png"
    ]
    if name in punct_names:
        return Truth.PUNCT
    
    # Everything else is a signature (content)
    return Truth.SIGN

def compare():
    data_dir = Path("data")
    poc = load_module("poc.py")
    poc_son = load_module("poc_son.py")
    
    image_paths = sorted(list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.PNG")))
    
    poc_results = {"correct": 0, "total": 0}
    poc_son_results = {"correct": 0, "total": 0}
    
    table_rows = []
    
    for img_path in image_paths:
        truth = get_ground_truth(img_path.name)
        
        # Test poc.py (Binary: CONTENT/EMPTY)
        res_poc = poc.analyze(str(img_path))
        is_poc_correct = False
        if res_poc.result.name == "EMPTY":
            is_poc_correct = (truth == Truth.EMPTY)
        elif res_poc.result.name == "CONTENT":
            is_poc_correct = (truth == Truth.SIGN or truth == Truth.PUNCT)
            
        if is_poc_correct: poc_results["correct"] += 1
        poc_results["total"] += 1
        
        # Test poc_son.py (Ternary: SIGN/PUNCT/EMPTY)
        res_son = poc_son.analyze(str(img_path))
        is_son_correct = (res_son.result.name == truth.name)
        
        if is_son_correct: poc_son_results["correct"] += 1
        poc_son_results["total"] += 1
        
        table_rows.append({
            "file": img_path.name,
            "truth": truth.value,
            "poc": f"{res_poc.result.name} {'✓' if is_poc_correct else '✗'}",
            "poc_son": f"{res_son.result.name} {'✓' if is_son_correct else '✗'}"
        })

    print("=" * 70)
    print(f"{'FILE':<25} | {'TRUTH':<8} | {'POC.PY (2-class)':<15} | {'POC_SON.PY (3-class)'}")
    print("-" * 70)
    for row in table_rows:
        # Only show mismatching or a few samples if too many
        if "✗" in row["poc"] or "✗" in row["poc_son"]:
             print(f"{row['file']:<25} | {row['truth']:<8} | {row['poc']:<15} | {row['poc_son']}")
    
    print("-" * 70)
    poc_acc = poc_results["correct"] / poc_results["total"] * 100
    son_acc = poc_son_results["correct"] / poc_son_results["total"] * 100
    
    print(f"POC.PY Accuracy (Binary)    : {poc_acc:.1f}% ({poc_results['correct']}/{poc_results['total']})")
    print(f"POC_SON.PY Accuracy (Ternary): {son_acc:.1f}% ({poc_son_results['correct']}/{poc_son_results['total']})")
    print("=" * 70)

if __name__ == "__main__":
    compare()
