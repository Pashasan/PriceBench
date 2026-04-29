"""
Triage: compute first-option selection rate per model across both orderings.

A model's "position balance" measures whether its choices are driven by content
or by slot position. Models with extreme bias (|rate - 0.5| > THRESHOLD) are
flagged as position-locked -- they express no coherent preferences and are
excluded from downstream interpretive analysis.

Outputs:
  web/data/triage.csv -- one row per model: first_option_rate, verdict
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
from config import MODELS

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                       'web', 'data', 'triage.csv')

# Distance from 50% that counts as "locked". 0.35 puts the cutoff at 15%/85%.
THRESHOLD = 0.35


def provider_of(name):
    n = name.lower()
    if n.startswith('gpt'):
        return 'OpenAI'
    if n.startswith('claude'):
        return 'Anthropic'
    if n.startswith('qwen'):
        return 'Alibaba'
    if n.startswith('gemma'):
        return 'Google'
    if n.startswith('llama'):
        return 'Meta'
    if n.startswith('phi'):
        return 'Microsoft'
    if n.startswith('mistral'):
        return 'Mistral'
    if n.startswith('deepseek'):
        return 'DeepSeek'
    return 'Other'


rows = []
for name in MODELS:
    orig_file, swap_file = MODELS[name]
    orig = pd.read_csv(orig_file)
    swap = pd.read_csv(swap_file)
    # The swap CSV stores choices already mapped back to the original reference
    # frame. In the orig prompt, Option A is the first-shown option; in the
    # swap prompt Option B (original A) is the first-shown. So the share of
    # first-shown picks is (orig A + swap B) / total.
    orig_valid = orig[orig['choice'].isin(['A', 'B'])]
    swap_valid = swap[swap['choice'].isin(['A', 'B'])]
    n = len(orig_valid) + len(swap_valid)
    if n == 0:
        rows.append(dict(model=name, provider=provider_of(name),
                         n=0, first_rate=float('nan'), imbalance=float('nan'),
                         verdict='no data'))
        continue
    first_shown = (orig_valid['choice'] == 'A').sum() + \
                  (swap_valid['choice'] == 'B').sum()
    first_rate = first_shown / n
    imbalance = abs(first_rate - 0.5)
    if imbalance > THRESHOLD:
        verdict = 'locked-primacy' if first_rate > 0.5 else 'locked-recency'
    else:
        verdict = 'engaged'
    rows.append(dict(
        model=name,
        provider=provider_of(name),
        n=int(n),
        first_rate=float(first_rate),
        imbalance=float(imbalance),
        verdict=verdict,
    ))

out = pd.DataFrame(rows).sort_values('first_rate')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
out.to_csv(OUT_CSV, index=False)

print(f'{"Model":<22s} {"Provider":<12s} {"First-opt":>10s}  Verdict')
print('-' * 62)
for _, r in out.iterrows():
    print(f'{r["model"]:<22s} {r["provider"]:<12s} '
          f'{r["first_rate"]*100:>9.1f}%  {r["verdict"]}')

n_engaged = (out['verdict'] == 'engaged').sum()
n_locked = out['verdict'].str.startswith('locked').sum()
print(f'\nEngaged: {n_engaged} / {len(out)}   Locked: {n_locked} / {len(out)}')
print(f'Saved {OUT_CSV}')
