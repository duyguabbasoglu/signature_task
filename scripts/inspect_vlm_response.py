import sys
import json
from pathlib import Path
import requests

# Use repo venv python when running
ENDPOINT = 'https://common-inference-apis.turkcelltech.ai/gpt-oss-120b/v1'
API_KEY = 'uavPCHhER6/EZnQFc2JafwjyqkcPE0oL6sowlCWsLGw='
MODEL = 'gpt-oss-120b'

headers = {
    'accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

payload = {
    'model': MODEL,
    'messages': [
        {'role': 'system', 'content': 'You are helpful assistant'},
        {'role': 'user', 'content': 'Hello, what is your status?'}
    ],
    'max_tokens': 50
}

resp = requests.post(f"{ENDPOINT}/chat/completions", headers=headers, json=payload, verify=False, timeout=30)
print('STATUS:', resp.status_code)
try:
    j = resp.json()
    print(json.dumps(j, indent=2, ensure_ascii=False))
except Exception as e:
    print('Failed to parse JSON:', e)
    print(resp.text)
