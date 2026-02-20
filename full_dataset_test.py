#!/usr/bin/env python3
"""
Comprehensive Full Dataset Test Suite
- Scan all images in data/ folder
- Apply advanced rule-based classifier
- Write results to vlm_full_results.csv
- Generate final accuracy report
"""

import sys
import csv
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

sys.path.insert(0, '.')
from classifier import extract_features, classify_rule_based, ClassResult, load_image_robust

# Expected classifications based on filename patterns
EXPECTED_MAP = {
    # Empty
    'empty_': 'EMPTY',

    'emp': 'EMPTY',  # emp.jpg, emp2.jpg (simple line drawings)

    # Punctuation
    'punct_': 'PUNCT',
    'dot': 'PUNCT',
    'circle': 'PUNCT',
    'line': 'PUNCT',
    'check': 'PUNCT',
    'square': 'PUNCT',
    'x': 'PUNCT',
    # Converted test images (IMG_180X_converted) - PUNCT
    'IMG_180': 'PUNCT',
    # Signatures
    'B-S-': 'SIGN',  # Signature format
    'H-S-': 'SIGN',  # Signature format
    'IMG_': 'SIGN',  # Other converted signatures
}

def get_expected_class(filename: str) -> str:
    """Infer expected class from filename."""
    fname_lower = filename.lower()
    
    for pattern, expected in EXPECTED_MAP.items():
        if pattern.lower() in fname_lower:
            return expected
    
    # HEIC UUID files: assume SIGN (real signatures)
    if filename.lower().endswith('.heic'):
        return 'SIGN'
    
    # Default: assume signature if not recognized
    return 'SIGN'


def main():
    data_path = Path('data')
    if not data_path.exists():
        print("ERROR: data/ folder not found")
        return
    
    # Find all image files
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.heic']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(data_path.glob(pattern))
    
    image_files.sort()
    
    if not image_files:
        print("ERROR: No image files found in data/")
        return
    
    print("=" * 120)
    print("FULL DATASET CLASSIFICATION TEST")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Images found: {len(image_files)}")
    print("=" * 120)
    print()
    
    # Results storage
    results = []
    correct = 0
    total = 0
    
    # Classification headers
    print(f"{'#':<4} {'Filename':<40} {'Expected':<10} {'Result':<12} {'Conf':<6} {'Reason':<40} {'Status':<6}")
    print("-" * 120)
    
    for idx, img_path in enumerate(image_files, 1):
        filename = img_path.name
        expected = get_expected_class(filename)
        
        try:
            # Load and classify
            img = load_image_robust(str(img_path))
            if img is None:
                print(f"{idx:<4} {filename:<40} {expected:<10} ERROR      -      {'Could not read':<40} ERROR")
                results.append({
                    'filename': filename,
                    'expected': expected,
                    'result': 'ERROR',
                    'confidence': 0.0,
                    'reason': 'Could not read',
                    'ink_ratio': 0.0,
                    'cc_count': 0,
                    'complexity': 0.0,
                    'correct': False
                })
                continue
            
            # Extract features and classify
            features = extract_features(img)
            result, confidence, reason = classify_rule_based(features)
            
            total += 1
            
            # Check if correct
            result_short = result.value[0]  # E/P/S
            expected_short = expected[0]    # E/P/S
            is_correct = result_short == expected_short
            
            if is_correct:
                correct += 1
            
            status = "OK" if is_correct else "XX"
            
            print(
                f"{idx:<4} {filename:<40} {expected:<10} {result.value:<12} "
                f"{confidence:>5.0%} {reason[:40]:<40} {status:<6}"
            )
            
            results.append({
                'filename': filename,
                'expected': expected,
                'result': result.value,
                'confidence': confidence,
                'reason': reason,
                'ink_ratio': features.ink_ratio,
                'cc_count': features.cc_count,
                'complexity': features.complexity_score,
                'skeleton_length': features.skeleton_length,
                'endpoints': features.endpoints_count,
                'branchpoints': features.branchpoints_count,
                'correct': is_correct
            })
            
        except Exception as e:
            print(f"{idx:<4} {filename:<40} {expected:<10} ERROR      -      {str(e)[:40]:<40} ERROR")
            results.append({
                'filename': filename,
                'expected': expected,
                'result': 'ERROR',
                'confidence': 0.0,
                'reason': str(e),
                'ink_ratio': 0.0,
                'cc_count': 0,
                'complexity': 0.0,
                'correct': False
            })
    
    print("-" * 120)
    print()
    
    # Summary
    if total > 0:
        accuracy = correct / total * 100
        print(f"ACCURACY: {correct}/{total} = {accuracy:.1f}%")
    else:
        accuracy = 0.0
        print("No valid files processed")
    
    print()
    
    # Breakdown by class
    by_class = {}
    for r in results:
        exp = r['expected']
        if exp not in by_class:
            by_class[exp] = {'total': 0, 'correct': 0}
        
        by_class[exp]['total'] += 1
        if r['correct']:
            by_class[exp]['correct'] += 1
    
    print("Accuracy by Class:")
    for cls in sorted(by_class.keys()):
        stats = by_class[cls]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {cls:<10} {stats['correct']}/{stats['total']} = {acc:>5.1f}%")
    
    print()
    print("=" * 120)
    
    # Write to CSV
    csv_path = Path('vlm_full_results.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'expected', 'result', 'confidence', 'reason',
            'ink_ratio', 'cc_count', 'complexity', 'skeleton_length',
            'endpoints', 'branchpoints', 'correct'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to: {csv_path}")
    print(f"Total: {total} files | Accuracy: {accuracy:.1f}%")
    
    # Return success/failure
    return accuracy >= 90.0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
