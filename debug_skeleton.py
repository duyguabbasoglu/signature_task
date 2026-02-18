#!/usr/bin/env python3
"""Debug skeleton metrics."""
import sys
from classifier import extract_features
import cv2
from pathlib import Path

test_files = [
    ('data/punct_x.png', 'X'),
    ('data/punct_check.png', 'CHECK'),
    ('data/empty_noise.png', 'NOISE'),
]

for fpath, label in test_files:
    path = Path(fpath)
    if not path.exists():
        print(f"SKIP: {fpath}")
        continue
    
    img = cv2.imread(str(path))
    features = extract_features(img)
    
    print(f"{label:<15} cc={features.cc_count} skel_len={features.skeleton_length:>5} "
          f"ep={features.endpoints_count} bp={features.branchpoints_count} "
          f"complexity={features.complexity_score:.2f}")
