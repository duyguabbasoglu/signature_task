import csv

errors = []
with open('vlm_full_results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['correct'].lower() == 'false' and row['expected'] == 'SIGN':
            errors.append({
                'file': row['filename'],
                'predicted': row['result'],
                'reason': row['reason'],
                'ink_ratio': float(row['ink_ratio']),
                'cc_count': int(row['cc_count']),
                'complexity': float(row['complexity']),
                'skeleton_len': int(row['skeleton_length'])
            })

print(f"❌ {len(errors)} SIGN olması gereken dosya yanlış sınıflandırıldı:\n")
for e in errors:
    print(f"  {e['file']}")
    print(f"    Reason: {e['reason']}")
    print(f"    ink={e['ink_ratio']:.3f}, cc={e['cc_count']}, complexity={e['complexity']:.1f}, skel={e['skeleton_len']}")
    print()

# Propose fix
if errors:
    print("\n💡 Problem ve Çözüm:")
    print("   Hepsi shape=SQUARE olarak detected → complexity=10.0 (high) varmış")
    print("   Fix: complexity > 2.0 ise SQUARE shape check'i atla, SIGN yap")
