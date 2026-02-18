import sys
sys.path.insert(0, '.')
from classifier import load_image_robust, extract_features
from pathlib import Path

img_path = 'data/IMG_1809_converted.png'
img = load_image_robust(img_path)
features = extract_features(img)

print(f"IMG_1809 detailed features:")
print(f"  is_circle: {features.is_circle}")
print(f"  is_square: {features.is_square}")
print(f"  is_dot: {features.is_dot}")
print(f"  is_line: {features.is_line}")
print(f"  is_x: {features.is_x}")
print(f"  is_check: {features.is_check}")
print(f"  ink_ratio: {features.ink_ratio:.4f}")
print(f"  cc_count: {features.cc_count}")
print(f"  skeleton_length: {features.skeleton_length}")
print(f"  complexity_score: {features.complexity_score:.2f}")
