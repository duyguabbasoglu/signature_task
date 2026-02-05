import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)

# Ensure both repository root and src are on sys.path so imports work
repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, repo_root)
sys.path.insert(0, str(Path(repo_root) / 'src'))

import poc
from bbox_detector.vlm_client import VLMClient

# === Configure these ===
ENDPOINT = 'https://common-inference-apis.test-turkcelltech.ai/gpt-oss-120b/v1'
API_KEY = 'uavPCHhER6/EZnQFc2JafwjyqkcPE0oL6sowlCWsLGw='
MODEL = 'gpt-oss-120b'
TIMEOUT = 30

# Instantiate client with provided credentials
vlm = VLMClient(endpoint=ENDPOINT, api_key=API_KEY, model=MODEL, timeout=TIMEOUT)

poc._vlm_client = vlm
poc._vlm_enabled = True

images = [
    'data/IMG_1807_converted.png',
    'data/IMG_1808_converted.png',
    'data/IMG_1809_converted.png',
]

for p in images:
    try:
        r = poc.analyze(p, use_vlm=True)
        print(f'{p}: {r.result.value} (used_vlm={r.used_vlm}) confidence={r.confidence}')
    except Exception as e:
        print(f'{p}: ERROR - {e}')
