import sys
from pathlib import Path
import json
import time

repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, repo_root)
sys.path.insert(0, str(Path(repo_root) / 'src'))

from bbox_detector.vlm_client import VLMClient
import cv2
import os

ENDPOINT = os.getenv('BBOX_LLM_ENDPOINT', 'https://common-inference-apis.test-turkcelltech.ai/gpt-oss-120b/v1')
API_KEY = os.getenv('BBOX_LLM_API_KEY', 'uavPCHhER6/EZnQFc2JafwjyqkcPE0oL6sowlCWsLGw=')
MODEL = os.getenv('BBOX_LLM_MODEL', 'gpt-oss-120b')

vc = VLMClient(endpoint=ENDPOINT, api_key=API_KEY, model=MODEL)

images = [
    'data/IMG_1807_converted.png',
    'data/IMG_1808_converted.png',
    'data/IMG_1809_converted.png',
]

out = {}
for img_path in images:
    print('Processing', img_path)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        out[img_path] = {'error': 'failed to load'}
        continue
    try:
        b64 = vc._encode_image(img)
        content, reasoning = vc._call_vlm(b64)
        out[img_path] = {'content': content, 'reasoning': reasoning}
    except Exception as e:
        out[img_path] = {'error': str(e)}
    time.sleep(0.2)

with open('vlm_capture_sample.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print('Saved vlm_capture_sample.json')
