import sys
import time
import json
import csv
from pathlib import Path

# Ensure repo root and src on path
repo_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, repo_root)
sys.path.insert(0, str(Path(repo_root) / 'src'))

import poc
from bbox_detector.vlm_client import VLMClient

import os

# Optional .env is supported if you have python-dotenv installed, but not required.
ENDPOINT = os.getenv('BBOX_LLM_ENDPOINT', 'https://common-inference-apis.turkcelltech.ai/gpt-oss-120b/v1')
API_KEY = os.getenv('BBOX_LLM_API_KEY', 'uavPCHhER6/EZnQFc2JafwjyqkcPE0oL6sowlCWsLGw=')
MODEL = os.getenv('BBOX_LLM_MODEL', 'gpt-oss-120b')

OUT_JSON = 'vlm_full_results.json'
OUT_CSV = 'vlm_full_results.csv'

def gather_images(data_dir='data', full=False):
    p = Path(data_dir)
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tif', '.TIF')
    files = sorted([x for x in p.iterdir() if x.suffix in exts])
    if full:
        return [str(x) for x in files]
    # default: only process converted PNGs (smaller, already suitable)
    sample = [x for x in files if x.name.startswith('IMG_') and x.suffix.lower() == '.png']
    return [str(x) for x in sample]

def main():
    print('Initializing VLM client...')
    vlm = VLMClient(endpoint=ENDPOINT, api_key=API_KEY, model=MODEL)
    poc._vlm_client = vlm
    poc._vlm_enabled = True

    full_flag = os.getenv('FULL_VLM_RUN', '0') in ('1', 'true', 'yes')
    images = gather_images('data', full=full_flag)
    if not images:
        print('No images found in data/ to process')
        return

    results = []
    for i, img in enumerate(images, 1):
        print(f'[{i}/{len(images)}] Processing {img}...')
        try:
            # load image and features
            gray = poc.load_image(img)
            bw = poc.binarize(gray)
            total_area, component_count, largest_area = poc.analyze_components(bw)

            # call poc.classify but force VLM usage by calling vlm client directly
            vlm_client = poc.get_vlm_client()

            vlm_label = None
            vlm_conf = None
            vlm_reasoning = None
            used_vlm = False

            try:
                vlm_res = vlm_client.classify(gray)
                vlm_reasoning = vlm_res.reasoning
                vlm_conf = vlm_res.confidence
                used_vlm = vlm_res.used_vlm
                if vlm_res.result.name == 'SIGNATURE':
                    vlm_label = 'SIGN'
                elif vlm_res.result.name == 'PUNCTUATION':
                    vlm_label = 'PUNCT'
                else:
                    vlm_label = None
            except Exception as e:
                # VLM failed for this image; fall back
                vlm_reasoning = str(e)

            # fallback classification (no VLM) if necessary
            if not used_vlm:
                result, confidence, shape_type, _ = poc.classify(total_area, component_count, bw, gray, use_vlm=False)
                out = {
                    'file': img,
                    'result': result.value,
                    'confidence': confidence,
                    'shape_type': shape_type,
                    'used_vlm': False,
                }
            else:
                out = {
                    'file': img,
                    'result': vlm_label or 'UNKNOWN',
                    'confidence': vlm_conf,
                    'shape_type': 'vlm',
                    'used_vlm': True,
                    'vlm_reasoning': vlm_reasoning,
                }

            results.append(out)

        except Exception as e:
            print(f'Error processing {img}: {e}')
            results.append({'file': img, 'error': str(e)})

        # gentle pause to avoid rate spikes
        time.sleep(0.2)

    # Save JSON
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save CSV (flatten keys)
    keys = set()
    for r in results:
        keys.update(r.keys())
    keys = sorted(keys)

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Summary
    used_vlm = sum(1 for r in results if isinstance(r, dict) and r.get('used_vlm'))
    total = len(results)
    print(f'Done. Processed {total} images, VLM used for {used_vlm}. Results saved to {OUT_JSON} and {OUT_CSV}')

if __name__ == '__main__':
    main()
