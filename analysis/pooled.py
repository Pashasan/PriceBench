"""
Pooled binary + ternary conditional-logit analysis.

Rationale. Conditional logit handles variable group sizes natively: each
"choice situation" is one group with 2 or 3 options and exactly one chosen.
Pooling is valid if the preference parameters beta are the same across
formats. That is testable:

  Restricted model:  V_j = X_j beta
  Unrestricted:      V_j = X_j beta + X_j * ternary * gamma
  LR test on gamma: if insignificant, pooling is valid and roughly doubles
  the effective sample.

We fit three models per engaged LLM:
  (1) pooled restricted -- one beta vector across both formats
  (2) binary only       -- existing benchmark
  (3) pooled with format interactions -- to run the LR test

Outputs:
  web/data/personality_pooled.csv    per-model pooled beta, SE, LR stat, p
  web/data/decile_pooled.csv         per-model decile coefs from pooled NP
  web/data/pooling_comparison.csv    side-by-side SE for binary vs pooled

Claude Haiku 4.5 has no ternary data, so its "pooled" result is
mechanically identical to binary-only; we flag it.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.discrete.conditional_models import ConditionalLogit
from config import (MODELS, N_DECILES, compute_decile_edges, extract_bed_size,
                    TASKS_FILE, TASK_ID_SWAP_OFFSET)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
TRIAGE = pd.read_csv(os.path.join(BASE, 'web', 'data', 'triage.csv'))
engaged = TRIAGE[TRIAGE['verdict'] == 'engaged']['model'].tolist()

# Deciles computed on the POOLED price distribution across binary + ternary
tasks_all = pd.read_csv(TASKS_FILE)
all_prices_full = []
for _, r in tasks_all.iterrows():
    for col in ['a_price', 'b_price', 'c_price']:
        if not pd.isna(r[col]):
            all_prices_full.append(float(r[col]))
all_prices = pd.Series(all_prices_full)
edges = np.percentile(all_prices, np.arange(0, 110, 10))
edges[0] -= 0.5
labels = [f'D{i}' for i in range(1, N_DECILES + 1)]
assigned = pd.cut(all_prices, bins=edges, labels=labels)
decile_medians = all_prices.groupby(assigned, observed=False).median().values

controls = ['stars', 'bed_rank', 'cancel', 'breakfast',
            'review_score', 'review_count_100']


def build_long_pooled(model_name):
    """Long-format dataset pooling binary (2-option) + ternary (3-option)
    groups for one LLM. group_id encodes (task_id, ordering) so the two
    orderings of the same task are separate groups."""
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
        for _, res in r.iterrows():
            tid = int(res['task_id'])
            choice = res['choice']
            t = tasks_all[tasks_all['task_id'] == tid]
            if t.empty:
                continue
            t = t.iloc[0]
            task_type = t['task_type']
            opts = ['a', 'b'] if task_type == 'binary' else ['a', 'b', 'c']
            valid = {'A', 'B'} if task_type == 'binary' else {'A', 'B', 'C'}
            if choice not in valid:
                continue
            gid = tid * 2 + ordering
            is_tern = 1 if task_type == 'ternary' else 0
            for opt in opts:
                rows.append(dict(
                    group_id=gid,
                    task_id=tid,
                    option=opt.upper(),
                    task_type=task_type,
                    is_tern=is_tern,
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
    # Keep only groups with exactly one chosen option
    ok = df.groupby('group_id')['chosen'].transform('sum') == 1
    df = df[ok]
    # Decile dummies on the pooled-price deciles
    df['decile'] = pd.cut(df['price'], bins=edges, labels=labels)
    for i in range(2, N_DECILES + 1):
        df[f'D{i}'] = (df['decile'] == f'D{i}').astype(int)
    return df


def fit_clogit(df, regressors):
    X = df[regressors].astype(float)
    y = df['chosen']
    g = df['group_id']
    return ConditionalLogit(y, X, groups=g).fit(
        disp=False, method='bfgs', maxiter=500)


pers_rows = []
dec_rows = []
cmp_rows = []

for name in engaged:
    df = build_long_pooled(name)
    if df.empty:
        print(f'SKIP {name}: no data')
        continue

    # Diagnostic: how many tasks of each type did we end up with
    n_bin_groups = (df[df['task_type'] == 'binary']['group_id'].nunique())
    n_ter_groups = (df[df['task_type'] == 'ternary']['group_id'].nunique())

    bin_df = df[df['task_type'] == 'binary']
    ter_df = df[df['task_type'] == 'ternary']

    # --- (1) binary-only baseline ---
    try:
        m_bin = fit_clogit(bin_df, controls + ['log_price'])
    except Exception as e:
        print(f'  {name} binary fail: {e}')
        continue

    # --- (2) ternary-only (may be missing for Claude) ---
    m_ter = None
    if n_ter_groups > 100:
        try:
            m_ter = fit_clogit(ter_df, controls + ['log_price'])
        except Exception as e:
            m_ter = None

    # --- (3) pooled restricted (one beta) ---
    try:
        m_pool = fit_clogit(df, controls + ['log_price'])
    except Exception as e:
        print(f'  {name} pooled fail: {e}')
        continue

    # --- (4) pooled unrestricted (interactions) ---
    # Interact every control with is_tern to allow format-specific slopes
    lr_p = np.nan
    lr_chi2 = np.nan
    if m_ter is not None:
        df_u = df.copy()
        interaction_cols = []
        for c in controls + ['log_price']:
            col = f'{c}_x_tern'
            df_u[col] = df_u[c] * df_u['is_tern']
            interaction_cols.append(col)
        regressors_full = controls + ['log_price'] + interaction_cols
        try:
            m_full = fit_clogit(df_u, regressors_full)
            # LR test (same groups, strictly nested)
            lr_chi2 = 2 * (m_full.llf - m_pool.llf)
            lr_p = float(chi2.sf(lr_chi2, len(interaction_cols)))
        except Exception as e:
            pass

    # --- (5) NP decile spec on pooled data ---
    decile_vars = [f'D{i}' for i in range(2, N_DECILES + 1)]
    try:
        m_np = fit_clogit(df, controls + decile_vars)
        for i in range(2, N_DECILES + 1):
            dv = f'D{i}'
            dec_rows.append(dict(
                model=name, decile=i,
                median_price=float(decile_medians[i - 1]),
                coef=float(m_np.params[dv]),
                se=float(m_np.bse[dv]),
            ))
        dec_rows.append(dict(model=name, decile=1,
                              median_price=float(decile_medians[0]),
                              coef=0.0, se=0.0))
    except Exception as e:
        print(f'  {name} NP fail: {e}')

    # Record results
    row = dict(
        model=name,
        n_bin_groups=int(n_bin_groups),
        n_ter_groups=int(n_ter_groups),
        b_logp_bin=float(m_bin.params['log_price']),
        se_logp_bin=float(m_bin.bse['log_price']),
        b_logp_pool=float(m_pool.params['log_price']),
        se_logp_pool=float(m_pool.bse['log_price']),
        b_logp_ter=float(m_ter.params['log_price']) if m_ter else np.nan,
        se_logp_ter=float(m_ter.bse['log_price']) if m_ter else np.nan,
        llf_bin=float(m_bin.llf),
        llf_pool=float(m_pool.llf),
        lr_chi2=float(lr_chi2),
        lr_df=7,
        lr_p=float(lr_p),
    )
    pers_rows.append(row)
    cmp_rows.append(dict(
        model=name,
        se_bin=row['se_logp_bin'],
        se_pool=row['se_logp_pool'],
        se_ratio=row['se_logp_pool'] / row['se_logp_bin'],
        b_bin=row['b_logp_bin'],
        b_pool=row['b_logp_pool'],
        lr_p=row['lr_p'],
        has_ternary=(m_ter is not None),
    ))
    print(f'  {name:<22s}  b_bin={row["b_logp_bin"]:+.3f}(se={row["se_logp_bin"]:.3f})  '
          f'b_pool={row["b_logp_pool"]:+.3f}(se={row["se_logp_pool"]:.3f})  '
          f'LR p={lr_p:.3g}')

pd.DataFrame(pers_rows).to_csv(
    os.path.join(BASE, 'web', 'data', 'personality_pooled.csv'), index=False)
pd.DataFrame(dec_rows).to_csv(
    os.path.join(BASE, 'web', 'data', 'decile_pooled.csv'), index=False)
pd.DataFrame(cmp_rows).to_csv(
    os.path.join(BASE, 'web', 'data', 'pooling_comparison.csv'), index=False)

# Summary
cmp = pd.DataFrame(cmp_rows)
print('\n=== Summary ===')
print(f'Mean SE ratio (pooled/binary): {cmp["se_ratio"].mean():.3f}  '
      f'(theoretical sqrt(1/2) = 0.707)')
print(f'Models with LR p < 0.05 (reject pooling): '
      f'{(cmp["lr_p"] < 0.05).sum()} / {cmp["has_ternary"].sum()}')
print(f'Models with LR p < 0.01: '
      f'{(cmp["lr_p"] < 0.01).sum()} / {cmp["has_ternary"].sum()}')
print('\nSaved personality_pooled.csv, decile_pooled.csv, '
      'pooling_comparison.csv')
