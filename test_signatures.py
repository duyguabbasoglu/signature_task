#!/usr/bin/env python3
"""Test to see if real signatures are no longer misclassified"""

from classifier import extract_features, classify_rule_based, load_image_robust

# Test a few real signatures that were previously misclassified as PUNCT by the sparse_marks rule
test_files = [
    ('data/001_02.PNG', 'SIGN'),
    ('data/001_05.PNG', 'SIGN'),
    ('data/001_06.PNG', 'SIGN'),
]

print("\n=== TESTING REAL SIGNATURES ===\n")
for filepath, expected in test_files:
    try:
        img = load_image_robust(filepath)
        feat = extract_features(img)
        result, conf, reason = classify_rule_based(feat)
        
        status = "✓" if result.value == expected else "✗"
        print(f"{status} {filepath.split('/')[-1]:30} Expected: {expected:10} Got: {result.value:10} ({conf:.0%})")
        print(f"   cc={feat.cc_count}, endpoints={feat.endpoints_count}, branchpoints={feat.branchpoints_count}")
        print(f"   skeleton={feat.skeleton_length}, ink={feat.ink_ratio:.4f}")
    except Exception as e:
        print(f"✗ {filepath:30} ERROR: {str(e)[:50]}")

print()
