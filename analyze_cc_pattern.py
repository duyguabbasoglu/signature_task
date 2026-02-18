import csv

# Analyze misclassified HEIC files
errors_heic = []
correct_heic = []
with open('vlm_full_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if 'heic' in row['filename'].lower():
            cc = int(row['cc_count'])
            complexity = float(row['complexity'])
            if row['correct'].lower() == 'false':
                errors_heic.append({
                    'file': row['filename'],
                    'expected': row['expected'],
                    'predicted': row['result'],
                    'cc': cc,
                    'complexity': complexity
                })
            else:
                correct_heic.append({'file': row['filename'], 'cc': cc, 'complexity': complexity})

print(f"HEIC Analysis:")
print(f"  Correct: {len(correct_heic)} files")
if correct_heic:
    cc_vals = [x['cc'] for x in correct_heic]
    print(f"    CC range: {min(cc_vals)} - {max(cc_vals)}")

print(f"\n  Incorrect: {len(errors_heic)} files")
if errors_heic:
    for e in errors_heic[:10]:
        print(f"    {e['file']}: cc={e['cc']}, expected={e['expected']}, predicted={e['predicted']}")
    cc_vals = [x['cc'] for x in errors_heic]
    print(f"    CC range: {min(cc_vals)} - {max(cc_vals)}")
