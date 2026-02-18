#!/usr/bin/env python3
"""Debug binarization issue."""
import cv2
import numpy as np
from pathlib import Path

test_files = [
    'data/punct_dot.png',
    'data/punct_x.png',
    'data/empty_white.png',
    'data/IMG_1807_converted.png'
]

for fpath in test_files:
    path = Path(fpath)
    if not path.exists():
        continue
    
    img = cv2.imread(str(path))
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Otsu binarize
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Count
    black_pixels = np.sum(binary_otsu == 255)
    ink_ratio = black_pixels / binary_otsu.size
    
    print(f"{path.name:<30} ink_ratio={ink_ratio:.4f} (black pixels={black_pixels})")
    
    # Show if > 90% black
    if ink_ratio > 0.9:
        # Try without invert
        _, binary_normal = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        black_pixels_normal = np.sum(binary_normal == 255)
        white_pixels_normal = np.sum(binary_normal == 0)
        print(f"  Without INVERT: black={black_pixels_normal}, white={white_pixels_normal}")
