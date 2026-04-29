"""
Personality metrics: per-model price sensitivity and WTP coefficients.

For each model we estimate two parametric logits (log-price and linear-price)
on the binary tasks, extract coefficients with standard errors, and derive:

  WTP per review point = -beta_review / beta_price      (dollars per 0.1)
  WTP per star         = -beta_stars  / beta_price
  WTP per cancellation = -beta_cancel / beta_price

Uses the log-price specification for WTP (elasticity interpretation).
For linear-price WTP we use the linear-price spec so the ratio is in dollars.

Outputs:
  web/data/personality.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import MODELS, build_dataset

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_CSV = os.path.join(BASE, 'web', 'data', 'personality.csv')
TRIAGE_CSV = os.path.join(BASE, 'web', 'data', 'triage.csv')

triage = pd.read_csv(TRIAGE_CSV).set_index('model')

CTRLS = ['d_stars', 'd_bed', 'd_cancel', 'd_breakfast',
        'd_review_score', 'd_review_count_100']


def fit(df, price_var):
    X = sm.add_constant(df[CTRLS + [price_var]])
    return sm.Logit(df['choice_A'], X).fit(disp=False)


rows = []
for name in MODELS:
    verdict = triage.loc[name, 'verdict'] if name in triage.index else 'no data'
    try:
        df = build_dataset(name, task_type='binary')
        if df['choice_A'].std() < 1e-6:
            raise RuntimeError('no variation in choices')
        m_log = fit(df, 'd_log_price')
        m_lin = fit(df, 'd_price')
    except Exception as e:
        rows.append(dict(model=name, verdict=verdict, error=str(e)))
        continue

    row = dict(model=name, verdict=verdict, error='',
               r2_log=m_log.prsquared, r2_lin=m_lin.prsquared,
               aic_log=m_log.aic, aic_lin=m_lin.aic,
               n=int(m_log.nobs))
    # price coefs from each spec
    row['b_logp']   = m_log.params['d_log_price']
    row['b_logp_se'] = m_log.bse['d_log_price']
    row['b_logp_p']  = m_log.pvalues['d_log_price']
    row['b_linp']   = m_lin.params['d_price']
    row['b_linp_se'] = m_lin.bse['d_price']
    row['b_linp_p']  = m_lin.pvalues['d_price']

    # quality coefs from log-price spec (preferred for interpretation)
    for v in CTRLS:
        row[f'b_{v}'] = m_log.params[v]
        row[f'b_{v}_se'] = m_log.bse[v]
        row[f'b_{v}_p'] = m_log.pvalues[v]

    # WTP: use linear-price spec so ratio is in dollars.
    # WTP(quality) = -beta(quality) / beta(price)
    bp = m_lin.params['d_price']
    if abs(bp) > 1e-6:
        row['wtp_star']      = -m_lin.params['d_stars'] / bp
        row['wtp_review']    = -m_lin.params['d_review_score'] / bp
        row['wtp_cancel']    = -m_lin.params['d_cancel'] / bp
        row['wtp_breakfast'] = -m_lin.params['d_breakfast'] / bp
    else:
        for k in ('wtp_star', 'wtp_review', 'wtp_cancel', 'wtp_breakfast'):
            row[k] = float('nan')

    # Functional form verdict (AIC-based)
    d_aic = m_lin.aic - m_log.aic
    if m_log.pvalues['d_log_price'] > 0.05 and m_lin.pvalues['d_price'] > 0.05:
        row['fit_form'] = 'insignificant'
    elif d_aic < -2:
        row['fit_form'] = 'linear'
    elif d_aic > 2:
        row['fit_form'] = 'log'
    else:
        row['fit_form'] = 'tie'

    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

# Print a compact summary (engaged only, sorted by |b_logp|)
eng = out[out['verdict'] == 'engaged'].copy()
eng['abs_blogp'] = eng['b_logp'].abs()
eng = eng.sort_values('abs_blogp', ascending=False)

print(f'{"Model":<22s} {"b_ln(p)":>9s} {"b_p":>9s} {"R2_log":>7s} {"R2_lin":>7s} '
      f'{"WTP/$":>7s} {"fit":>8s}')
print('-' * 78)
for _, r in eng.iterrows():
    print(f'{r["model"]:<22s} {r["b_logp"]:>9.2f} {r["b_linp"]*100:>9.3f} '
          f'{r["r2_log"]:>7.3f} {r["r2_lin"]:>7.3f} '
          f'{r["wtp_star"]:>7.1f} {r["fit_form"]:>8s}')
print(f'\nSaved {OUT_CSV}')
