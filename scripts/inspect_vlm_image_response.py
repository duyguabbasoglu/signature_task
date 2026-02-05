import base64
import json
from pathlib import Path
import requests

ENDPOINT = 'https://common-inference-apis.test-turkcelltech.ai/gpt-oss-120b/v1'
API_KEY = 'uavPCHhER6/EZnQFc2JafwjyqkcPE0oL6sowlCWsLGw='
MODEL = 'gpt-oss-120b'
IMAGE_PATH = 'data/IMG_1807_converted.png'

with open(IMAGE_PATH, 'rb') as f:
    b = f.read()

img_b64 = base64.b64encode(b).decode('utf-8')

headers = {
    'accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

messages = [
    {'role': 'system', 'content': "You are an image classification expert.\nRespond with ONLY one word: SIGNATURE or PUNCTUATION"},
    {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'What does this bounding box image contain? Classify it as either SIGNATURE or PUNCTUATION.'},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img_b64}'}}
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
