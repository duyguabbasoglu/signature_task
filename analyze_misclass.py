import pandas as pd

df = pd.read_csv('vlm_full_results.csv')

# Misclassified SIGN as PUNCT
sign_as_punct = df[(df['expected'] == 'SIGN') & (df['result'] == 'PUNCT')]
print(f'SIGN misclassified as PUNCT: {len(sign_as_punct)} cases')
print(f'Avg confidence: {sign_as_punct["confidence"].mean():.1%}')
print(f'Avg ink_ratio: {sign_as_punct["ink_ratio"].mean():.3f}')
print(f'Avg cc_count: {sign_as_punct["cc_count"].mean():.1f}')
print(f'Avg complexity: {sign_as_punct["complexity"].mean():.2f}')
print(f'Avg skeleton_length: {sign_as_punct["skeleton_length"].mean():.0f}')
print()

# Correct SIGN classifications (for comparison)
sign_correct = df[(df['expected'] == 'SIGN') & (df['result'] == 'SIGN')]
print(f'SIGN correct: {len(sign_correct)} cases')
print(f'Avg complexity: {sign_correct["complexity"].mean():.2f}')
print(f'Avg cc_count: {sign_correct["cc_count"].mean():.1f}')
print(f'Avg skeleton_length: {sign_correct["skeleton_length"].mean():.0f}')
