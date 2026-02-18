from PIL import Image
import numpy as np

for fname in ['IMG_1807_converted.png', 'IMG_1808_converted.png', 'IMG_1809_converted.png']:
    img = Image.open(f'data/{fname}')
    arr = np.array(img)
    print(f"\n{fname}:")
    print(f"  Size: {img.size}")
    print(f"  Mode: {img.mode}")
    print(f"  Shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
