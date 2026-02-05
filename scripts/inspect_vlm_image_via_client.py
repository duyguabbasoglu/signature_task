import sys
from pathlib import Path

# Ensure repo root and src on path
repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, repo_root)
sys.path.insert(0, str(Path(repo_root) / 'src'))

from bbox_detector.vlm_client import VLMClient
import requests
import json
import cv2
import numpy as np

ENDPOINT = 'https://common-inference-apis.test-turkcelltech.ai/gpt-oss-120b/v1'
API_KEY = 'uavPCHhER6/EZnQFc2JafwjyqkcPE0oL6sowlCWsLGw='
MODEL = 'gpt-oss-120b'
IMG = 'data/IMG_1807_converted.png'

vc = VLMClient(endpoint=ENDPOINT, api_key=API_KEY, model=MODEL)
img = cv2.imread(IMG, cv2.IMREAD_UNCHANGED)
if img is None:
    raise SystemExit('Failed to load image')

img_b64 = vc._encode_image(img)
print('encoded length:', len(img_b64))

headers = {
    'accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Build payload as client does
messages = [
    {'role': 'system', 'content': vc.SYSTEM_PROMPT},
    {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': vc.USER_PROMPT},
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpg;base64,{img_b64}'}}
        ]
    }
]

payload = {
    'model': MODEL,
    'messages': messages,
    'max_tokens': 50,
    'temperature': 0.1
}

resp = requests.post(f"{ENDPOINT}/chat/completions", headers=headers, json=payload, verify=False, timeout=60)
print('STATUS:', resp.status_code)
try:
    j = resp.json()
    print(json.dumps(j, indent=2, ensure_ascii=False))
except Exception as e:
    print('Failed to parse JSON:', e)
    print(resp.text)
