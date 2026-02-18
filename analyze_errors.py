#!/usr/bin/env python3
"""
Advanced rule-based classification test with feature analysis.
Tests the new classifier.py with detailed metrics.
"""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')
from classifier import extract_features, classify_rule_based

files = [
    ('data/empty_white.png', 'EMPTY', 'white background'),
    ('data/empty_black.png', 'EMPTY', 'full black'),
    ('data/empty_noise.png', 'EMPTY', 'small noise'),
    ('data/punct_dot.png', 'PUNCT', 'simple dot'),
    ('data/punct_x.png', 'PUNCT', 'X mark'),
    ('data/punct_circle.png', 'PUNCT', 'circle'),
    ('data/punct_line.png', 'PUNCT', 'line'),
    ('data/punct_check.png', 'PUNCT', 'check mark'),
    ('data/punct_square.png', 'PUNCT', 'square'),
    ('data/IMG_1807_converted.png', 'SIGN', 'signature 1'),
    ('data/IMG_1808_converted.png', 'SIGN', 'signature 2'),
    ('data/IMG_1809_converted.png', 'PUNCT', 'pseudo-signature'),
]

print("=" * 120)
print("ADVANCED RULE-BASED CLASSIFICATION TEST")
print("=" * 120)
print()
print(f"{'File':<30} {'Expected':<8} {'Result':<10} {'Conf':<6} {'Complexity':<8} {'Ink':<6} {'Status':<6} Reason")
print("-" * 120)

correct = 0
total = 0

for fpath, expected, description in files:
    path = Path(fpath)
    
    if not path.exists():
        print(f"{path.name:<30} {expected:<8} SKIP       -      -        -      SKIP   FILE NOT FOUND")
        continue
    
    try:
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path.name:<30} {expected:<8} ERROR      -      -        -      ERROR  Could not read")
            continue
        
        # Extract features
        features = extract_features(img)
        
        # Classify
        result, confidence, reason = classify_rule_based(features)
        
        total += 1
        
        # Check if correct (map result to expected)
        result_short = result.value[0]  # E/P/S/A
        expected_short = expected[0]    # E/P/S
        
        is_correct = result_short == expected_short
        if is_correct:
            correct += 1
        
        status = "OK" if is_correct else "XX"  # Simple ASCII
        
        print(
            f"{path.name:<30} {expected:<8} {result.value:<10} "
            f"{confidence:>5.0%} {features.complexity_score:>7.2f} "
            f"{features.ink_ratio:>5.3f} {status:<6} {reason[:40]}"
        )
        
    except Exception as e:
        print(f"{path.name:<30} {expected:<8} ERROR      -      -        -      ERROR  {str(e)[:40]}")

print("-" * 120)
if total > 0:
    accuracy = correct / total * 100
    print(f"Accuracy: {correct}/{total} = {accuracy:.1f}%")
print("=" * 120)
