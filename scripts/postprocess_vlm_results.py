import json
import csv
from pathlib import Path

INPUT_JSON_CANDIDATES = [
    Path('vlm_capture_sample.json'),
    Path('vlm_full_results.json')
]

OUT_JSON = Path('vlm_processed.json')
OUT_CSV = Path('vlm_processed.csv')

def extract_label(text: str) -> str:
    if not text:
        return ''
    t = text.upper()
    if 'SIGNATURE' in t:
        return 'SIGNATURE'
    if 'PUNCTUATION' in t or 'PUNCT' in t:
        return 'PUNCTUATION'
    return ''

def load_input():
    for p in INPUT_JSON_CANDIDATES:
        if p.exists():
            with p.open('r', encoding='utf-8') as f:
                return json.load(f), p
    raise FileNotFoundError('No input JSON found (vlm_capture_sample.json or vlm_full_results.json)')

def main():
    data, src = load_input()
    print(f'Loaded {len(data)} entries from {src}')

    processed = []
    for k, v in data.items() if isinstance(data, dict) else enumerate(data):
        # support dict-of-files or list-of-records
        if isinstance(data, dict):
            file = k
            record = v
        else:
            record = v
            file = record.get('file') or record.get('name') or f'item_{k}'

        content = ''
        reasoning = ''
        # possible shapes
        if isinstance(record, dict):
            content = record.get('content') or record.get('result') or ''
            reasoning = record.get('reasoning') or record.get('reason') or ''

        label = extract_label(content) or extract_label(reasoning)

        used_vlm = bool(label)

        out = {
            'file': file,
            'llm_label': label if label else None,
            'used_vlm': used_vlm,
            'raw_content': content,
            'raw_reasoning': reasoning,
        }

        # include any original keys if available
        if isinstance(record, dict):
            for kk, vv in record.items():
                if kk in out:
                    continue
                out[kk] = vv

        processed.append(out)

    # save json
    with OUT_JSON.open('w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    # save csv
    keys = set()
    for r in processed:
        keys.update(r.keys())
    keys = sorted(keys)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in processed:
            writer.writerow({k: (r.get(k) if r.get(k) is not None else '') for k in keys})

    # summary
    total = len(processed)
    used_vlm = sum(1 for r in processed if r.get('used_vlm'))
    print(f'Processed {total} entries, VLM label found for {used_vlm}')

if __name__ == '__main__':
    main()
