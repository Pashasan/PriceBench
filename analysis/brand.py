"""
Brand-FE analysis for engaged models.

Estimates a logit with per-brand difference dummies (Independent = reference)
on top of the usual controls + price deciles. Extracts brand coefficient
vectors per model, builds the model-by-brand matrix, and tests whether
same-provider models have more similar brand preferences than cross-provider
pairs via a permutation test on mean pairwise Pearson correlation.

Outputs:
  web/data/brand_coefs.csv   -- long-form (model, brand, coef, se, p)
  web/data/brand_matrix.csv  -- wide model x brand matrix
  web/data/brand_cluster.json -- within vs cross mean r and permutation p-value
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, pearsonr
from config import MODELS, N_DECILES, build_dataset, compute_decile_edges
from analysis.triage import provider_of

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
TRIAGE = pd.read_csv(os.path.join(BASE, 'web', 'data', 'triage.csv')).set_index('model')

BRAND_KEYWORDS = {
    'Hilton':  ['Hampton Inn', 'Hilton Garden Inn', 'DoubleTree', 'Conrad',
                'Embassy Suites', 'Tempo by Hilton', 'Motto by Hilton',
                'New York Hilton', 'Ink 48'],
    'Marriott':['Courtyard by Marriott', 'Fairfield Inn', 'Residence Inn',
                'SpringHill Suites', 'AC Hotel', 'Renaissance', 'Sheraton',
                'Westin', 'W New York', 'Moxy', 'Aloft', 'Marriott',
                'Ritz-Carlton', 'EDITION', 'St. Regis', 'Lotte'],
    'IHG':     ['Holiday Inn', 'Crowne Plaza', 'Hotel Indigo', 'Even Hotel',
                'EVEN Hotel', 'InterContinental', 'Kimpton', 'voco'],
    'Hyatt':   ['Hyatt', 'Park Hyatt', 'Thompson', 'Andaz'],
    'Wyndham': ['Days Inn', 'La Quinta', 'Ramada', 'Wyndham', 'Best Western'],
}
BRANDS = list(BRAND_KEYWORDS.keys())


def get_brand(name):
    for brand, kws in BRAND_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in name.lower():
                return brand
    return 'Independent'


engaged = TRIAGE[TRIAGE['verdict'] == 'engaged'].index.tolist()
edges = compute_decile_edges()
rows = []

for name in engaged:
    print(f'  fitting {name}...')
    try:
        df = build_dataset(name, decile_edges=edges)
    except Exception as e:
        print(f'    SKIP: {e}')
        continue
    df['a_brand'] = df['a_name'].apply(get_brand)
    df['b_brand'] = df['b_name'].apply(get_brand)

    brand_vars = []
    for b in BRANDS:
        v = f'd_brand_{b}'
        df[v] = ((df['a_brand'] == b).astype(int)
                 - (df['b_brand'] == b).astype(int))
        brand_vars.append(v)

    decile_vars = [f'dd_D{i}' for i in range(2, N_DECILES + 1)]
    controls = ['d_stars', 'd_bed', 'd_cancel', 'd_breakfast',
                'd_review_score', 'd_review_count_100']
    X = sm.add_constant(df[controls + brand_vars + decile_vars])
    try:
        m = sm.Logit(df['choice_A'], X).fit(disp=False, maxiter=200)
    except Exception as e:
        print(f'    LOGIT FAIL: {e}')
        continue
    for b in BRANDS:
        v = f'd_brand_{b}'
        rows.append(dict(
            model=name, provider=provider_of(name), brand=b,
            coef=float(m.params[v]), se=float(m.bse[v]),
            pvalue=float(m.pvalues[v]),
        ))

coefs = pd.DataFrame(rows)
coefs.to_csv(os.path.join(BASE, 'web', 'data', 'brand_coefs.csv'), index=False)

# Wide matrix model x brand
mat = coefs.pivot(index='model', columns='brand', values='coef')
mat = mat.reindex(columns=BRANDS)
mat.to_csv(os.path.join(BASE, 'web', 'data', 'brand_matrix.csv'))

# Provider clustering via permutation test on mean within-provider r
models_in_mat = list(mat.index)
prov = {m: provider_of(m) for m in models_in_mat}
vecs = {m: mat.loc[m].values for m in models_in_mat}


def pair_correlations(labels):
    """Mean correlation within same-label and across-label pairs."""
    within, across = [], []
    ms = list(vecs.keys())
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = ms[i], ms[j]
            va, vb = vecs[a], vecs[b]
            if np.std(va) < 1e-9 or np.std(vb) < 1e-9:
                continue
            r, _ = pearsonr(va, vb)
            if labels[a] == labels[b]:
                within.append(r)
            else:
                across.append(r)
    return (np.mean(within) if within else float('nan'),
            np.mean(across) if across else float('nan'),
            len(within), len(across))


true_within, true_across, nw, na = pair_correlations(prov)
true_gap = true_within - true_across

np.random.seed(42)
perm_gaps = []
for _ in range(2000):
    shuffled_providers = list(prov.values())
    np.random.shuffle(shuffled_providers)
    lab = dict(zip(prov.keys(), shuffled_providers))
    w, a, _, _ = pair_correlations(lab)
    perm_gaps.append(w - a)
perm_gaps = np.array(perm_gaps)
p_perm = float((perm_gaps >= true_gap).mean())

result = dict(
    n_engaged_models=len(models_in_mat),
    n_providers=len(set(prov.values())),
    within_provider_mean_r=float(true_within),
    across_provider_mean_r=float(true_across),
    gap=float(true_gap),
    permutation_p=p_perm,
    n_within_pairs=nw,
    n_across_pairs=na,
    providers=sorted(set(prov.values())),
)
with open(os.path.join(BASE, 'web', 'data', 'brand_cluster.json'), 'w') as f:
    json.dump(result, f, indent=2)

print(f'\nBrand clustering results:')
print(f'  engaged models          : {len(models_in_mat)}')
print(f'  within-provider mean r  : {true_within:+.3f}  ({nw} pairs)')
print(f'  across-provider mean r  : {true_across:+.3f}  ({na} pairs)')
print(f'  gap                     : {true_gap:+.3f}')
print(f'  permutation p (one-sided) : {p_perm:.4f}')
