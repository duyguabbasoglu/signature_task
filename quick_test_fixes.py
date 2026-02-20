#!/usr/bin/env python3
"""Quick test of the four problematic files"""

from classifier import extract_features, classify_rule_based, load_image_robust

test_files = [
    ('data/emp.jpg', 'EMPTY'),
    ('data/emp2.jpg', 'EMPTY'),
    ('data/IMG_1807_converted.png', 'PUNCT'),
    ('data/IMG_1808_converted.png', 'PUNCT'),
    ('data/IMG_1809_converted.png', 'PUNCT'),
]

print("\n=== TESTING FIXES ===\n")
for filepath, expected in test_files:
    try:
        img = load_image_robust(filepath)
        feat = extract_features(img)
        result, conf, reason = classify_rule_based(feat)
        
        status = "✓" if result.value == expected else "✗"
        print(f"{status} {filepath.split('/')[-1]:30} Expected: {expected:10} Got: {result.value:10} ({conf:.0%}) - {reason[:50]}")
    except Exception as e:
        print(f"✗ {filepath:30} ERROR: {str(e)[:50]}")

print()
