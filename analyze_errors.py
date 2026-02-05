#!/usr/bin/env python3
import sys
import cv2
import numpy as np

sys.path.insert(0, 'src')
import poc

files = [
    ('data/IMG_1807_converted.png', 'should be PUNCT'),
    ('data/IMG_1808_converted.png', 'should be PUNCT'),
    ('data/IMG_1809_converted.png', 'correct as PUNCT'),
    ('data/empty_noise.png', 'should be EMPTY'),
    ('data/punct_x.png', 'should be PUNCT')
]

print("File Analysis:")
print("-" * 100)
print(f"{'File':<30} {'Area':<8} {'Comps':<6} {'Largest':<8} {'Current':<15} {'Shape':<20} Expected")
print("-" * 100)

for fpath, note in files:
    try:
        gray = poc.load_image(fpath)
        bw = poc.binarize(gray)
        total_area, comp_count, largest = poc.analyze_components(bw)
        result, conf, shape, _ = poc.classify(total_area, comp_count, bw, gray, use_vlm=False)
        
        fname = fpath.split("/")[1]
        print(f"{fname:<30} {total_area:<8} {comp_count:<6} {largest:<8} {result.value:<15} {shape:<20} {note}")
    except Exception as e:
        print(f"{fpath}: ERROR {e}")
