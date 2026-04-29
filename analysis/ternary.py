"""
Ternary conditional logit for PriceBench.

For each engaged model, fit:
  - Log-price spec   (V = beta_lnp * ln(price) + controls)
  - Linear-price spec (V = beta_p * price + controls)
  - Non-parametric decile-dummy spec (D1 omitted)

Grouping: statsmodels ConditionalLogit needs each group to represent one
choice situation with exactly one option chosen. Here each ternary task is
scored in two orderings, so we form a synthetic group_id = (task_id,
ordering) to keep the two orderings separate. Triple fixed effects drop out
analytically within each group.

Outputs:
  web/data/personality_ternary.csv  -- per-model coefficients
  web/data/decile_ternary.csv       -- model x decile coefs for plotting
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import warnings
from statsmodels.discrete.conditional_models import ConditionalLogit
from config import (MODELS, N_DECILES, compute_decile_edges, extract_bed_size,
                    TASKS_FILE, TASK_ID_SWAP_OFFSET)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_PERS = os.path.join(BASE, 'web', 'data', 'personality_ternary.csv')
OUT_DEC = os.path.join(BASE, 'web', 'data', 'decile_ternary.csv')

TRIAGE = pd.read_csv(os.path.join(BASE, 'web', 'data', 'triage.csv'))
engaged = TRIAGE[TRIAGE['verdict'] == 'engaged']['model'].tolist()

edges = compute_decile_edges(task_type='ternary')
labels = [f'D{i}' for i in range(1, N_DECILES + 1)]

# Decile medians computed from actual prices in each decile bin
tasks_all = pd.read_csv(TASKS_FILE)
tasks_ter = tasks_all[tasks_all['task_type'] == 'ternary'].copy()
all_prices = pd.concat([tasks_ter['a_price'],
                         tasks_ter['b_price'],
                         tasks_ter['c_price']])
assigned = pd.cut(all_prices, bins=edges, labels=labels)
decile_medians = all_prices.groupby(assigned, observed=False).median().values

controls = ['stars', 'bed_rank', 'cancel', 'breakfast',
            'review_score', 'review_count_100']


def build_ternary_long(model_name):
    """Build long-format ternary dataset with per-ordering group ids."""
    orig_file, swap_file = MODELS[model_name]
    rows = []
    for path, ordering in [(orig_file, 0), (swap_file, 1)]:
        try:
            r = pd.read_csv(path)
        except FileNotFoundError:
            continue
        if ordering == 1:
            r = r.copy()
            r['task_id'] = r['task_id'] - TASK_ID_SWAP_OFFSET
        r = r[r['task_id'].between(1801, 3600)]  # ternary range
        for _, res in r.iterrows():
            tid = int(res['task_id'])
            task = tasks_ter[tasks_ter['task_id'] == tid]
            if task.empty:
                continue
            t = task.iloc[0]
            choice = res['choice']
            if choice not in ('A', 'B', 'C'):
                continue
            gid = tid * 2 + ordering
            for opt in ('a', 'b', 'c'):
                rows.append(dict(
                    group_id=gid,
                    task_id=tid,
                    option=opt.upper(),
                    chosen=int(choice == opt.upper()),
                    price=float(t[f'{opt}_price']),
                    log_price=float(np.log(t[f'{opt}_price'])),
                    stars=float(t[f'{opt}_stars']),
                    bed_rank=float(extract_bed_size(t[f'{opt}_room_type'])),
                    cancel=float(t[f'{opt}_cancellation_free']),
                    breakfast=float(t[f'{opt}_breakfast_included']),
                    review_score=float(t[f'{opt}_review_score']),
                    review_count_100=float(t[f'{opt}_review_count']) / 100,
                ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['decile'] = pd.cut(df['price'], bins=edges, labels=labels)
    for i in range(2, N_DECILES + 1):
        df[f'D{i}'] = (df['decile'] == f'D{i}').astype(int)
    return df


pers_rows = []
dec_rows = []

for name in engaged:
    df = build_ternary_long(name)
    if df.empty or df['chosen'].sum() == 0:
        print(f'SKIP {name}: no ternary data')
        continue

    n_groups = df['group_id'].nunique()
    complete = (df.groupby('group_id')['chosen'].sum() == 1).sum()
    if complete < 100:
        print(f'SKIP {name}: {complete}/{n_groups} groups with exactly 1 chosen')
        continue

    keep = df.groupby('group_id')['chosen'].transform('sum') == 1
    df = df[keep]
    groups = df['group_id']

    try:
        X_log = df[controls + ['log_price']].astype(float)
        m_log = ConditionalLogit(df['chosen'], X_log, groups=groups).fit(
            disp=False, method='bfgs', maxiter=500)
        X_lin = df[controls + ['price']].astype(float)
        m_lin = ConditionalLogit(df['chosen'], X_lin, groups=groups).fit(
            disp=False, method='bfgs', maxiter=500)
        dec_vars = [f'D{i}' for i in range(2, N_DECILES + 1)]
        X_np = df[controls + dec_vars].astype(float)
        m_np = ConditionalLogit(df['chosen'], X_np, groups=groups).fit(
            disp=False, method='bfgs', maxiter=500)
    except Exception as e:
        print(f'  {name} ESTIMATION FAIL: {e}')
        continue

    row = dict(
        model=name,
        n=int(m_log.nobs),
        n_groups=int(df['group_id'].nunique()),
        b_logp=float(m_log.params['log_price']),
        b_logp_se=float(m_log.bse['log_price']),
        b_logp_p=float(m_log.pvalues['log_price']),
        b_linp=float(m_lin.params['price']),
        b_linp_se=float(m_lin.bse['price']),
        b_linp_p=float(m_lin.pvalues['price']),
        llf_log=float(m_log.llf),
        llf_lin=float(m_lin.llf),
    )
    for v in controls:
        row[f'b_{v}'] = float(m_log.params[v])

    # Both specs have same k, so AIC comparison reduces to LL comparison.
    d_ll = m_log.llf - m_lin.llf
    if row['b_logp_p'] > 0.05 and row['b_linp_p'] > 0.05:
        row['fit_form'] = 'insignificant'
    elif d_ll > 1.0:
        row['fit_form'] = 'log'
    elif d_ll < -1.0:
        row['fit_form'] = 'linear'
    else:
        row['fit_form'] = 'tie'
    pers_rows.append(row)

    dec_rows.append(dict(model=name, decile=1,
                          median_price=float(decile_medians[0]),
                          coef=0.0, se=0.0))
    for i in range(2, N_DECILES + 1):
        v = f'D{i}'
        dec_rows.append(dict(
            model=name, decile=i,
            median_price=float(decile_medians[i - 1]),
            coef=float(m_np.params[v]),
            se=float(m_np.bse[v]),
        ))
    print(f'  {name:<22s} b_lnp={row["b_logp"]:+.3f} '
          f'(p={row["b_logp_p"]:.2g})  best={row["fit_form"]}  '
          f'n_groups={row["n_groups"]}')

pd.DataFrame(pers_rows).to_csv(OUT_PERS, index=False)
pd.DataFrame(dec_rows).to_csv(OUT_DEC, index=False)
print(f'\nSaved {OUT_PERS}')
print(f'Saved {OUT_DEC}')
