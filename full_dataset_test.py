#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

sys.path.insert(0, '.')
from classifier import extract_features, classify_rule_based, ClassResult, load_image_robust

EXPECTED_MAP = {
    'empty_': 'EMPTY',
    'emp': 'EMPTY',      # bos/cizgi
    'punct_': 'PUNCT',
    'dot': 'PUNCT',
    'circle': 'PUNCT',
    'line': 'PUNCT',
    'check': 'PUNCT',
    'square': 'PUNCT',
    'x': 'PUNCT',
    'IMG_180': 'PUNCT',  # heic punctlar
    'B-S-': 'SIGN',      
    'H-S-': 'SIGN',      
    'IMG_': 'SIGN',      # heic imzalar
}

def get_expected_class(filename: str) -> str:
    fname_lower = filename.lower()
    for pattern, expected in EXPECTED_MAP.items():
        if pattern.lower() in fname_lower:
            return expected
    if filename.lower().endswith('.heic'):
        return 'SIGN'
    return 'SIGN'

def main():
    data_path = Path('data')
    if not data_path.exists():
        print("hata: data/ yok")
        return
    
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.heic']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(data_path.glob(pattern))
    image_files.sort()
    
    if not image_files:
        print("hata: resim yok")
        return
    
    results = []
    correct = 0
    total = 0
    
    for idx, img_path in enumerate(image_files, 1):
        filename = img_path.name
        expected = get_expected_class(filename)
        try:
            img = load_image_robust(str(img_path))
            if img is None:
                raise ValueError("okunamadi")
            
            features = extract_features(img)
            result, confidence, reason = classify_rule_based(features)
            
            total += 1
            is_correct = result.value[0] == expected[0]
            if is_correct:
                correct += 1
            
            status = "OK" if is_correct else "XX"
            print(f"{filename[:20]:<20} {expected:<10} {result.value:<10} {status}")
            
            results.append({
                'filename': filename, 'expected': expected, 'result': result.value,
                'confidence': confidence, 'reason': reason, 'correct': is_correct,
                'ink_ratio': features.ink_ratio, 'cc_count': features.cc_count,
                'complexity': features.complexity_score, 'skeleton_length': features.skeleton_length,
                'endpoints': features.endpoints_count, 'branchpoints': features.branchpoints_count
            })
        except Exception as e:
            print(f"hata {filename}: {e}")
            
    if total > 0:
        print(f"\nbasari: {correct}/{total} = {correct/total*100:.1f}%")
        
    csv_path = Path('vlm_full_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    return (correct/total*100) >= 90.0 if total else False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
