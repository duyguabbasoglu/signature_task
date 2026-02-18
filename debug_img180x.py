import sys
sys.path.insert(0, '.')
from classifier import load_image_robust, extract_features, classify_rule_based
from pathlib import Path

files = ['data/IMG_1807_converted.png', 'data/IMG_1808_converted.png', 'data/IMG_1809_converted.png']

for fpath in files:
    img = load_image_robust(fpath)
    if img is None:
        print(f"ERROR: {fpath} - Could not load")
        continue
    
    features = extract_features(img)
    result, conf, reason = classify_rule_based(features)
    
    print(f"\n{'='*60}")
    print(f"FILE: {Path(fpath).name}")
    print(f"  Predicted: {str(result)} ({conf*100:.0f}%) - {reason}")
    print(f"  Features: ink={features.ink_ratio:.4f}, cc={features.cc_count}, complexity={features.complexity_score:.2f}")


