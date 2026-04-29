"""
Generate all PriceBench web figures from the data CSVs in web/data/.

Figures (web/figures/):
  fig_triage.png         -- first-option rates per model, engaged vs locked
  fig_decile_curves.png  -- non-parametric price response, engaged only, shared y
  fig_quality_heatmap.png -- standardized quality coefs, engaged
  fig_personality.png    -- scatter: price sensitivity vs quality sensitivity
  fig_brand_heatmap.png  -- model x brand coefficients, engaged
  fig_funform.png        -- log vs linear overlay on non-parametric, shared y
                            (only models with genuine negative price response)
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FixedLocator, FuncFormatter

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA = os.path.join(BASE, 'web', 'data')
FIG  = os.path.join(BASE, 'web', 'figures')
os.makedirs(FIG, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

PROVIDER_COLORS = {
    'OpenAI': '#10A37F',
    'Anthropic': '#D97757',
    'Google': '#4285F4',
    'Meta': '#0866FF',
    'Alibaba': '#FF6A00',
    'Microsoft': '#5E5CE6',
    'Mistral': '#FA520F',
    'DeepSeek': '#4D6BFE',
    'Other': '#888',
}

triage = pd.read_csv(os.path.join(DATA, 'triage.csv'))
personality = pd.read_csv(os.path.join(DATA, 'personality.csv'))
deciles = pd.read_csv(os.path.join(DATA, 'decile_pooled.csv'))
brands = pd.read_csv(os.path.join(DATA, 'brand_matrix.csv'), index_col=0)

engaged = triage[triage['verdict'] == 'engaged']['model'].tolist()
ENGAGED_SET = set(engaged)

# Shared axis helpers ---------------------------------------------------------
PRICE_TICKS = [100, 200, 400, 800]


def _format_price(v, _pos=None):
    if v >= 1000:
        return f'${v/1000:.0f}k'
    return f'${v:.0f}'


def _configure_log_x(ax):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(PRICE_TICKS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_price))


# ============================================================================
# Figure 1: Triage
# ============================================================================
def fig_triage():
    df = triage.dropna(subset=['first_rate']).sort_values('first_rate')
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = []
    for _, r in df.iterrows():
        if r['verdict'] == 'engaged':
            colors.append(PROVIDER_COLORS.get(r['provider'], '#888'))
        else:
            colors.append('#bbbbbb')
    y = np.arange(len(df))
    bars = ax.barh(y, df['first_rate'] * 100, color=colors,
                   edgecolor='white', height=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels(df['model'])
    ax.axvline(50, color='#222', lw=1.0, linestyle='-')
    ax.axvspan(0, 15, color='#f0f0f0', zorder=0)
    ax.axvspan(85, 100, color='#f0f0f0', zorder=0)

    # Locked-band labels placed to the OUTSIDE of the bands so they don't
    # overlap bars or risk being misread as belonging to any single row.
    ax.text(7.5, len(df) + 0.7, 'locked\n(recency)',
            ha='center', va='bottom', color='#666', fontsize=9)
    ax.text(92.5, len(df) + 0.7, 'locked\n(primacy)',
            ha='center', va='bottom', color='#666', fontsize=9)
    # Make room at the top of the axes for those labels
    ax.set_ylim(-0.5, len(df) + 2.4)

    for bar, (_, r) in zip(bars, df.iterrows()):
        x = bar.get_width()
        ax.text(x + 1.5 if x < 90 else x - 1.5, bar.get_y() + bar.get_height()/2,
                f'{x:.1f}%', va='center',
                ha='left' if x < 90 else 'right',
                fontsize=8.5, color='#333')

    ax.set_xlim(0, 100)
    ax.set_xlabel('First-shown selection rate (%) — pooled across orderings')
    ax.text(50, -1.35, '50% = balanced', ha='center', color='#444',
            fontsize=9)
    ax.set_title('Triage: which models actually engage with content?',
                 loc='left', fontweight='bold', fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_triage.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Figure 2: Non-parametric decile curves (engaged, shared y)
# ============================================================================
def fig_decile_curves():
    eng = personality[personality['verdict'] == 'engaged'].copy()
    eng['abs_b'] = eng['b_logp'].abs()
    eng = eng.sort_values('abs_b', ascending=False)
    models_to_plot = [m for m in eng['model'].tolist() if m in ENGAGED_SET]

    n = len(models_to_plot)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4),
                              sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    # Compute shared y limits so all panels are directly comparable
    sub_eng = deciles[deciles['model'].isin(models_to_plot)]
    y_lo = (sub_eng['coef'] - sub_eng['se']).min() - 0.3
    y_hi = (sub_eng['coef'] + sub_eng['se']).max() + 0.3

    triage_idx = triage.set_index('model')
    for i, name in enumerate(models_to_plot):
        ax = axes[i // cols, i % cols]
        sub = deciles[deciles['model'] == name].sort_values('decile')
        if sub.empty:
            ax.set_axis_off()
            continue
        x = sub['median_price'].values
        y = sub['coef'].values
        se = sub['se'].values
        prov = triage_idx.loc[name, 'provider']
        color = PROVIDER_COLORS.get(prov, '#666')
        ax.plot(x, y, '-o', color=color, lw=1.7, ms=3.8)
        ax.fill_between(x, y - se, y + se, color=color, alpha=0.18, lw=0)
        ax.axhline(0, color='#aaa', lw=0.6, ls=':')
        ax.set_title(name, fontsize=10, pad=2)
        r2 = eng[eng['model'] == name]['r2_log'].values
        if len(r2):
            ax.text(0.97, 0.04, f'R²={r2[0]:.2f}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8, color='#666')
        _configure_log_x(ax)
        ax.tick_params(labelsize=8)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_axis_off()

    for c in range(cols):
        axes[-1, c].set_xlabel('Price (log scale)', fontsize=9)
    for r in range(rows):
        axes[r, 0].set_ylabel('Utility vs cheapest decile', fontsize=9)

    for ax in axes.flat:
        if ax.has_data():
            ax.set_ylim(y_lo, y_hi)

    fig.suptitle('Non-parametric price response — all 23 engaged models, '
                 'shared y-axis, sorted by |β_ln(p)|',
                 fontsize=12, fontweight='bold', y=1.005)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_decile_curves.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Figure 3: Quality sensitivity heatmap
# ============================================================================
def fig_quality_heatmap():
    eng = personality[personality['verdict'] == 'engaged'].copy()
    eng = eng.sort_values('b_logp', ascending=True).reset_index(drop=True)
    cols = ['b_d_review_score', 'b_d_stars', 'b_d_cancel',
            'b_d_breakfast', 'b_d_bed']
    labels = ['ΔReview', 'ΔStars', 'ΔCancel', 'ΔBreakfast', 'ΔBed']
    M = eng[cols].values.astype(float)
    M_z = (M - M.mean(0)) / (M.std(0) + 1e-9)

    fig, ax = plt.subplots(figsize=(8.5, max(5, 0.32 * len(eng))))
    im = ax.imshow(M_z, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(eng)))
    ax.set_yticklabels(eng['model'], fontsize=9)
    for i in range(len(eng)):
        for j in range(len(cols)):
            v = M[i, j]
            shade = 'white' if abs(M_z[i, j]) > 1.2 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color=shade)
    ax.set_title('Quality sensitivity per model (raw logit coefs, '
                 'cells colored by within-attribute z-score)',
                 fontsize=11, loc='left', fontweight='bold', pad=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('z-score across models', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_quality_heatmap.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Figure 4: Personality map
# ============================================================================
def _nudge_labels(positions, min_dy=0.18):
    order = sorted(range(len(positions)),
                   key=lambda i: (positions[i][0], positions[i][1]))
    out = list(positions)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        dx = abs(out[i][0] - out[j][0])
        dy = abs(out[i][1] - out[j][1])
        if dx < 0.15 and dy < min_dy:
            shift = min_dy - dy + 0.02
            out[j] = (out[j][0], out[j][1] + shift)
    return out


def fig_personality():
    eng = personality[(personality['verdict'] == 'engaged')
                       & ((personality['b_logp_p'] < 0.05)
                          | (personality['b_linp_p'] < 0.05))].copy()
    eng['abs_b'] = eng['b_logp'].abs()
    eng['quality'] = eng['b_d_review_score']

    fig, ax = plt.subplots(figsize=(11, 8))
    triage_idx = triage.set_index('model')
    xs = eng['abs_b'].tolist()
    ys = eng['quality'].tolist()
    label_pos = _nudge_labels(list(zip(xs, ys)))

    for (x, y), (lx, ly), (_, r) in zip(zip(xs, ys), label_pos, eng.iterrows()):
        prov = triage_idx.loc[r['model'], 'provider']
        color = PROVIDER_COLORS.get(prov, '#666')
        ax.scatter(x, y, s=140, color=color, edgecolor='white', lw=1.2,
                   alpha=0.92, zorder=3)
        ax.annotate(r['model'], (lx, ly), xytext=(8, 0),
                    textcoords='offset points', fontsize=9, color='#222',
                    va='center')

    ax.set_xlabel('Price sensitivity  |β_ln(p)|   '
                  '(higher = more price-averse, holding quality fixed)',
                  fontsize=11)
    ax.set_ylabel('Quality sensitivity  β_review_score   '
                  '(higher = more quality-driven, holding price fixed)',
                  fontsize=11)
    ax.axhline(0, color='#aaa', lw=0.6, ls=':')

    xmid = float(eng['abs_b'].median())
    ymid = float(eng['quality'].median())
    ax.axvline(xmid, color='#bbb', lw=0.5, ls='--')
    ax.axhline(ymid, color='#bbb', lw=0.5, ls='--')

    # Give the axes some headroom so quadrant labels have space ABOVE the data
    x_lo, x_hi = ax.get_xlim(); y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + (y_hi - y_lo) * 0.10)
    ax.set_xlim(x_lo - (x_hi - x_lo) * 0.02, x_hi + (x_hi - x_lo) * 0.02)

    # Quadrant labels placed at the top as annotations above the data area
    # so they don't overlap with points or the legend.
    ax.set_title(
        'PriceBench personality map — engaged models with significant price effect',
        loc='left', fontweight='bold', fontsize=13, pad=32)
    quad_kw = dict(fontsize=10, color='#777', fontstyle='italic',
                   transform=ax.transAxes)
    ax.text(xmid / ax.get_xlim()[1] / 2 + 0.005, 1.02,
            'lets price slide + chases quality', ha='left', **quad_kw)
    ax.text(1.0 - (xmid / ax.get_xlim()[1]) * 0.0 - 0.005, 1.02,
            'sharp on price + chases quality', ha='right', **quad_kw)

    # Lower quadrant hints at the bottom of the data area, tucked against axes
    ax.text(0.005, 0.01, 'indifferent / weak price signal',
            ha='left', va='bottom', fontsize=10, color='#aaa',
            fontstyle='italic', transform=ax.transAxes)

    # Provider legend — place OUTSIDE data area (to the right) so it never
    # overlaps points.
    seen, handles = set(), []
    order_providers = ['OpenAI', 'Anthropic', 'Google', 'Meta', 'Alibaba',
                        'Microsoft', 'Mistral', 'DeepSeek']
    for prov in order_providers:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=PROVIDER_COLORS[prov],
                       markersize=10, label=prov))
    ax.legend(handles=handles, loc='center left',
              bbox_to_anchor=(1.005, 0.5), frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_personality.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Figure 5: Brand heatmap
# ============================================================================
def fig_brand_heatmap():
    M = brands.copy()
    order = M.abs().sum(axis=1).sort_values(ascending=True).index
    M = M.loc[order]
    fig, ax = plt.subplots(figsize=(7.5, max(5, 0.32 * len(M))))
    vmax = max(abs(M.values.min()), abs(M.values.max()))
    im = ax.imshow(M.values, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels(M.columns, fontsize=10)
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels(M.index, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            color = 'white' if abs(v) > 0.6 * vmax else 'black'
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                    fontsize=7.5, color=color)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('logit coef (vs Independent)', fontsize=8)
    ax.set_title('Brand preferences after controlling for price + quality + amenities\n'
                 '(positive = preferred over an Independent at same controls)',
                 fontsize=10.5, loc='left', fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_brand_heatmap.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Figure 6: Log vs linear vs nonparametric
# ============================================================================
def fig_funform():
    """
    Filter to models with a genuine negative price response -- i.e. the
    log-price coefficient is negative and statistically significant at p<0.05
    AND the non-parametric D10 coefficient is meaningfully below zero (< -0.3).
    This excludes models like Gemma3 1B (positive coef) and Phi-2 2.7B (flat)
    where the log-vs-linear question is not well posed.
    """
    d_last = deciles.sort_values('decile').groupby('model').tail(1)[
        ['model', 'coef']
    ].rename(columns={'coef': 'd10_coef'})
    merged = personality.merge(d_last, on='model', how='left')

    eng = merged[(merged['verdict'] == 'engaged')
                  & (merged['b_logp'] < 0)
                  & (merged['b_logp_p'] < 0.05)
                  & (merged['d10_coef'] < -0.3)].copy()
    eng['abs_b'] = eng['b_logp'].abs()
    eng = eng.sort_values('abs_b', ascending=False)
    models_to_plot = eng['model'].tolist()

    n = len(models_to_plot)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4),
                              sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    # Shared y range over these models
    sub = deciles[deciles['model'].isin(models_to_plot)]
    y_lo = (sub['coef'] - sub['se']).min() - 0.5
    y_hi = (sub['coef'] + sub['se']).max() + 0.5

    for i, name in enumerate(models_to_plot):
        ax = axes[i // cols, i % cols]
        ss = deciles[deciles['model'] == name].sort_values('decile')
        x = ss['median_price'].values
        y = ss['coef'].values
        se = ss['se'].values
        ax.fill_between(x, y - se, y + se, color='#888', alpha=0.15, lw=0)
        ax.plot(x, y, 'o', ms=4.5, color='#222')

        row = eng[eng['model'] == name].iloc[0]
        x_ref = x[0]
        log_pred = row['b_logp'] * (np.log(x) - np.log(x_ref))
        lin_pred = row['b_linp'] * (x - x_ref)
        best = row['fit_form']

        # Winning curve bold; losing curve pale.
        LIN_STYLE = dict(color='#E89A2F', ls='--')
        LOG_STYLE = dict(color='#7B5BBE')
        if best == 'linear':
            ax.plot(x, lin_pred, lw=2.0, **LIN_STYLE)
            ax.plot(x, log_pred, lw=1.0, alpha=0.35, **LOG_STYLE)
        elif best == 'log':
            ax.plot(x, log_pred, lw=2.0, **LOG_STYLE)
            ax.plot(x, lin_pred, lw=1.0, alpha=0.35, **LIN_STYLE)
        else:
            ax.plot(x, log_pred, lw=1.5, **LOG_STYLE)
            ax.plot(x, lin_pred, lw=1.5, **LIN_STYLE)

        d_aic = row['aic_lin'] - row['aic_log']  # negative = linear wins
        marker = best if best != 'tie' else 'tie'
        ax.set_title(f'{name}   [best: {marker}]', fontsize=9, pad=2)
        ax.text(0.97, 0.04, f'ΔAIC={d_aic:+.0f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7.5, color='#777')
        ax.axhline(0, color='#bbb', lw=0.5, ls=':')
        _configure_log_x(ax)
        ax.tick_params(labelsize=8)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_axis_off()
    for c in range(cols):
        axes[-1, c].set_xlabel('Price (log scale)', fontsize=9)
    for r in range(rows):
        axes[r, 0].set_ylabel('Utility vs D1', fontsize=9)

    for ax in axes.flat:
        if ax.has_data():
            ax.set_ylim(y_lo, y_hi)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#222',
                   markersize=6, label='non-parametric (decile dummies)'),
        plt.Line2D([0], [0], color='#7B5BBE', lw=2,
                   label='log-price fit'),
        plt.Line2D([0], [0], color='#E89A2F', ls='--', lw=2,
                   label='linear-price fit'),
    ]
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Log vs linear price fit — winner bold, loser pale. '
                 'Shared y-axis.',
                 fontsize=12, fontweight='bold', y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_funform.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Print the filter result for reporting
    fit_counts = eng['fit_form'].value_counts().to_dict()
    print(f'funform figure: {n} models passed filter; fit counts = {fit_counts}')


# ============================================================================
# Figure 7: Binary vs ternary forest plot (log + linear price)
# ============================================================================
def fig_binary_vs_ternary():
    """Two-panel forest plot: log-price beta (left) and linear-price beta
    (right), each with binary and ternary estimates per engaged model.
    Models are sorted on a SHARED y-axis by binary log-price coefficient
    from most-negative (top) to least."""
    bin_df = pd.read_csv(os.path.join(DATA, 'personality.csv'))
    ter_df = pd.read_csv(os.path.join(DATA, 'personality_ternary.csv'))
    bin_ = bin_df[bin_df['verdict'] == 'engaged'][
        ['model', 'b_logp', 'b_logp_se', 'b_linp', 'b_linp_se']
    ].rename(columns={'b_logp': 'b_logp_bin', 'b_logp_se': 'se_logp_bin',
                       'b_linp': 'b_linp_bin', 'b_linp_se': 'se_linp_bin'})
    ter = ter_df[['model', 'b_logp', 'b_logp_se', 'b_linp', 'b_linp_se']].rename(
        columns={'b_logp': 'b_logp_ter', 'b_logp_se': 'se_logp_ter',
                 'b_linp': 'b_linp_ter', 'b_linp_se': 'se_linp_ter'})
    m = bin_.merge(ter, on='model', how='left')
    m = m.sort_values('b_logp_bin', ascending=True).reset_index(drop=True)
    # most-negative first row; invert y-axis later so it ends up on top

    from scipy.stats import spearmanr, pearsonr
    valid = m['b_logp_ter'].notna()
    rho_log, _ = spearmanr(m.loc[valid, 'b_logp_bin'].abs(),
                            m.loc[valid, 'b_logp_ter'].abs())
    r_log, _ = pearsonr(m.loc[valid, 'b_logp_bin'],
                         m.loc[valid, 'b_logp_ter'])
    rho_lin, _ = spearmanr(m.loc[valid, 'b_linp_bin'].abs(),
                            m.loc[valid, 'b_linp_ter'].abs())
    r_lin, _ = pearsonr(m.loc[valid, 'b_linp_bin'],
                         m.loc[valid, 'b_linp_ter'])

    fig, (ax_log, ax_lin) = plt.subplots(
        1, 2, figsize=(12.5, 8), sharey=True,
        gridspec_kw={'wspace': 0.08})

    y = np.arange(len(m))
    BIN_COLOR = '#2C5F8A'
    TER_COLOR = '#C4631D'

    # --- Log panel ---
    for i, row in m.iterrows():
        if not pd.isna(row['b_logp_ter']):
            ax_log.plot([row['b_logp_bin'], row['b_logp_ter']],
                        [y[i], y[i]], '-', color='#ccc', lw=0.8, zorder=1)
    ax_log.errorbar(m['b_logp_bin'], y, xerr=m['se_logp_bin'],
                     fmt='o', color=BIN_COLOR, ms=6, capsize=2,
                     label='binary', zorder=3)
    ax_log.errorbar(m.loc[valid, 'b_logp_ter'], y[valid],
                     xerr=m.loc[valid, 'se_logp_ter'],
                     fmt='s', color=TER_COLOR, ms=5.5, capsize=2,
                     label='ternary', zorder=3)
    ax_log.axvline(0, color='#bbb', lw=0.6, ls=':')
    ax_log.set_yticks(y)
    ax_log.set_yticklabels(m['model'], fontsize=9)
    ax_log.invert_yaxis()
    ax_log.set_xlabel(r'$\beta_{\ln(p)}$  log-price coefficient',
                       fontsize=10.5)
    ax_log.set_title(
        f'Log-price   (Spearman ρ = {rho_log:.2f}, Pearson r = {r_log:.2f})',
        fontsize=10.5, loc='left', pad=8)
    ax_log.legend(loc='lower left', frameon=False, fontsize=9)

    # --- Linear panel ---
    for i, row in m.iterrows():
        if not pd.isna(row['b_linp_ter']):
            ax_lin.plot([row['b_linp_bin'], row['b_linp_ter']],
                        [y[i], y[i]], '-', color='#ccc', lw=0.8, zorder=1)
    ax_lin.errorbar(m['b_linp_bin'], y, xerr=m['se_linp_bin'],
                     fmt='o', color=BIN_COLOR, ms=6, capsize=2,
                     zorder=3)
    ax_lin.errorbar(m.loc[valid, 'b_linp_ter'], y[valid],
                     xerr=m.loc[valid, 'se_linp_ter'],
                     fmt='s', color=TER_COLOR, ms=5.5, capsize=2,
                     zorder=3)
    ax_lin.axvline(0, color='#bbb', lw=0.6, ls=':')
    ax_lin.set_xlabel(r'$\beta_{p}$  linear-price coefficient (per-dollar)',
                       fontsize=10.5)
    ax_lin.set_title(
        f'Linear price   (Spearman ρ = {rho_lin:.2f}, Pearson r = {r_lin:.2f})',
        fontsize=10.5, loc='left', pad=8)

    fig.suptitle(
        'Binary vs ternary price coefficients per model '
        '(models sorted by binary log-price, shared y-axis)',
        fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_binary_vs_ternary.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Figure 8: Pooling comparison - binary, ternary, pooled betas per model
# ============================================================================
def fig_pooling():
    cmp = pd.read_csv(os.path.join(DATA, 'pooling_comparison.csv'))
    pool = pd.read_csv(os.path.join(DATA, 'personality_pooled.csv'))
    ter = pd.read_csv(os.path.join(DATA, 'personality_ternary.csv'))
    bin_ = pd.read_csv(os.path.join(DATA, 'personality.csv'))

    m = cmp.merge(
        ter[['model', 'b_logp', 'b_logp_se']].rename(
            columns={'b_logp': 'b_ter', 'b_logp_se': 'se_ter'}),
        on='model', how='left').merge(
        bin_[['model', 'b_logp_se']].rename(columns={'b_logp_se': 'se_bin_full'}),
        on='model', how='left').merge(
        pool[['model', 'se_logp_pool']].rename(columns={'se_logp_pool': 'se_pool_full'}),
        on='model', how='left')

    # Sort by binary |beta|, descending so sharpest at top
    m['abs_bin'] = m['b_bin'].abs()
    m = m.sort_values('abs_bin', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(m))

    # Each model gets three points with error bars
    ax.errorbar(m['b_bin'], y - 0.22, xerr=m['se_bin'], fmt='o',
                color='#2C5F8A', ms=6, capsize=2, label='binary only')
    valid_ter = m['b_ter'].notna()
    ax.errorbar(m.loc[valid_ter, 'b_ter'], y[valid_ter],
                xerr=m.loc[valid_ter, 'se_ter'], fmt='s',
                color='#C4631D', ms=5, capsize=2, label='ternary only')
    ax.errorbar(m['b_pool'], y + 0.22, xerr=m['se_pool'], fmt='D',
                color='#0E7A4F', ms=5, capsize=2, label='pooled (restricted)')

    ax.axvline(0, color='#bbb', lw=0.6, ls=':')
    ax.set_yticks(y)
    ax.set_yticklabels(m['model'], fontsize=9)
    ax.set_xlabel(r'$\beta_{\ln(p)}$  (log-price coefficient, with ±1 SE)')
    ax.set_title('Binary, ternary, and pooled estimates per model',
                 loc='left', fontweight='bold', fontsize=12, pad=18)

    # Flag models where LR test rejects pooling
    rejected = (m['lr_p'] < 0.01).sum()
    total = m['has_ternary'].sum()
    ax.text(0.01, 1.01,
            f'LR test rejects restricted pooling at p<0.01 for {rejected}/{total} models',
            transform=ax.transAxes, fontsize=9.5, color='#b54430',
            fontstyle='italic', va='bottom')

    ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_pooling.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================================
# Figure 2d: Parameter count (capability) vs |beta_ln(p)|
# ============================================================================
# Open-weight parameter counts (billions, best public estimate).
# For MoE models we use ACTIVE parameters since that drives compute.
# Proprietary model counts are not public; we use rough tier-based estimates
# that approximately match API pricing ladders, and flag them as estimates.
MODEL_PARAMS_B = {
    # Open-weight (known)
    'Qwen3 0.6B': 0.6, 'Qwen3 1.7B': 1.7, 'Qwen3 4B': 4.0, 'Qwen3 8B': 8.0,
    'Qwen3 30B-A3B': 3.0,  # MoE, 3B active
    'Gemma3 1B': 1.0, 'Gemma3 4B': 4.0, 'Gemma3 12B': 12.0, 'Gemma3 27B': 27.0,
    'Gemma2 9B': 9.0,
    'Llama3.2 1B': 1.0, 'Llama3.2 3B': 3.0, 'Llama3 8B': 8.0,
    'Llama3.1 8B': 8.0,
    'Phi-2 2.7B': 2.7, 'Phi-3 Mini': 3.8, 'Phi-3 Medium': 14.0,
    'Phi-4 Mini': 3.8, 'Phi-4 14B': 14.0,
    'Mistral 7B': 7.0, 'Mistral-Nemo 12B': 12.0,
    'DeepSeek-R1 1.5B': 1.5, 'DeepSeek-R1 7B': 7.0,
    # Proprietary (ESTIMATES from pricing tiers)
    'GPT-4.1 Nano': 8.0,   'GPT-4.1 Mini': 40.0,
    'GPT-5.4 Nano': 10.0,  'GPT-5.4 Mini': 60.0,  'GPT-5.4': 400.0,
    'Claude Haiku 4.5': 25.0,
}
PROPRIETARY = {'GPT-4.1 Nano', 'GPT-4.1 Mini', 'GPT-5.4 Nano',
               'GPT-5.4 Mini', 'GPT-5.4', 'Claude Haiku 4.5'}


def fig_scaling():
    """Scatter: model size vs binary log-price sensitivity among engaged models.
    Filled circles = open-weight (real size), open squares = proprietary
    (tier-based size estimate). Within-family connecting lines."""
    p = pd.read_csv(os.path.join(DATA, 'personality.csv'))
    e = p[(p['verdict'] == 'engaged') & (p['b_logp_p'] < 0.05)].copy()
    e['params_b'] = e['model'].map(MODEL_PARAMS_B)
    e = e.dropna(subset=['params_b']).copy()
    e['absb'] = e['b_logp'].abs()
    e['is_prop'] = e['model'].isin(PROPRIETARY)

    from scipy.stats import pearsonr, spearmanr
    open_e = e[~e['is_prop']]
    r_open, p_open = pearsonr(np.log10(open_e['params_b']), open_e['absb'])
    rho_open, _ = spearmanr(np.log10(open_e['params_b']), open_e['absb'])
    r_all, _ = pearsonr(np.log10(e['params_b']), e['absb'])

    fig, ax = plt.subplots(figsize=(10.5, 7))
    triage_idx = triage.set_index('model')

    # Within-family connecting lines
    families = {
        'Qwen3': ['Qwen3 4B', 'Qwen3 8B', 'Qwen3 30B-A3B'],
        'Gemma3': ['Gemma3 1B', 'Gemma3 4B', 'Gemma3 12B', 'Gemma3 27B'],
        'Phi-3': ['Phi-3 Mini', 'Phi-3 Medium'],
        'Phi-4': ['Phi-4 Mini', 'Phi-4 14B'],
        'DeepSeek-R1': ['DeepSeek-R1 1.5B', 'DeepSeek-R1 7B'],
        'GPT-4.1': ['GPT-4.1 Nano', 'GPT-4.1 Mini'],
        'GPT-5.4': ['GPT-5.4 Nano', 'GPT-5.4 Mini', 'GPT-5.4'],
    }
    for fam, members in families.items():
        sub = e[e['model'].isin(members)].sort_values('params_b')
        if len(sub) < 2:
            continue
        prov = triage_idx.loc[sub.iloc[0]['model'], 'provider']
        color = PROVIDER_COLORS.get(prov, '#888')
        ax.plot(sub['params_b'], sub['absb'], '-',
                color=color, lw=1.0, alpha=0.22, zorder=1)

    # Points
    for _, r in e.iterrows():
        prov = triage_idx.loc[r['model'], 'provider']
        color = PROVIDER_COLORS.get(prov, '#888')
        if r['is_prop']:
            ax.scatter(r['params_b'], r['absb'], s=110, marker='s',
                       facecolor='white', edgecolor=color, lw=1.8, zorder=3)
        else:
            ax.scatter(r['params_b'], r['absb'], s=130, marker='o',
                       color=color, edgecolor='white', lw=1.2, zorder=3)

    # Custom per-model label offsets. Defaults go up-and-right; models in the
    # crowded bottom-left get bespoke positions to avoid overlap.
    DEFAULT_OFFSET = (8, 5)
    LABEL_OFFSET = {
        'DeepSeek-R1 1.5B': (8, 8),      # right, above (point at x-axis edge)
        'Llama3.2 3B':      (8, -13),    # right, below
        'Qwen3 30B-A3B':    (8, -14),    # right, below
        'Phi-3 Mini':       (-8, -12),   # left, below
        'Phi-4 Mini':       (8, 10),     # right, above
        'Qwen3 8B':         (8, -13),    # right, below
        'DeepSeek-R1 7B':   (8, 8),      # right, above
        'Qwen3 4B':         (-40, 5),    # left of point
        'Gemma3 4B':        (-35, 8),    # left-above
        'Mistral-Nemo 12B': (-60, 5),    # pull left
        'Gemma3 12B':       (8, -12),    # right, below
        'Phi-3 Medium':     (8, 5),      # right, above
        'Gemma2 9B':        (8, -4),     # right center
        'GPT-5.4 Nano':     (8, 8),
        'GPT-4.1 Nano':     (-60, -12),  # left-below to avoid GPT-5.4 Nano
    }
    HALIGN = {  # horizontal alignment override when nudging left
        'Qwen3 4B': 'right', 'Gemma3 4B': 'right', 'Mistral-Nemo 12B': 'right',
        'Phi-3 Mini': 'right',
        'GPT-4.1 Nano': 'right',
    }
    for _, r in e.iterrows():
        off = LABEL_OFFSET.get(r['model'], DEFAULT_OFFSET)
        ha = HALIGN.get(r['model'], 'left')
        ax.annotate(r['model'], (r['params_b'], r['absb']),
                    xytext=off, textcoords='offset points',
                    fontsize=8.5, color='#333', ha=ha, va='center')

    ax.set_xscale('log')
    ax.set_xlabel('Parameter count (B, log scale).   '
                  'Squares = proprietary tier-based estimates.',
                  fontsize=10.5)
    ax.set_ylabel(r'|$\beta_{\ln(p)}$|   (binary log-price sensitivity)',
                   fontsize=10.5)
    ax.set_title(
        f'Capability (parameter count) vs price sensitivity   '
        f'(open-weight only: r = {r_open:.2f}, ρ = {rho_open:.2f}, '
        f'n = {len(open_e)})',
        loc='left', fontweight='bold', fontsize=12, pad=12)

    # Provider legend
    seen, handles = set(), []
    for _, r in e.iterrows():
        prov = triage_idx.loc[r['model'], 'provider']
        if prov in seen:
            continue
        seen.add(prov)
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=PROVIDER_COLORS[prov],
                       markersize=10, label=prov))
    ax.legend(handles=handles, loc='upper left', frameon=False,
              fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_scaling.png'), dpi=170,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'  open-weight n={len(open_e)}  Pearson r={r_open:.3f} '
          f'(p={p_open:.3g})  Spearman rho={rho_open:.3f}')
    print(f'  all-engaged n={len(e)}  Pearson r={r_all:.3f}')


if __name__ == '__main__':
    print('triage...');              fig_triage()
    print('decile curves...');       fig_decile_curves()
    print('quality heatmap...');     fig_quality_heatmap()
    print('personality...');         fig_personality()
    print('brand heatmap...');       fig_brand_heatmap()
    print('functional form...');     fig_funform()
    print('binary vs ternary...');   fig_binary_vs_ternary()
    print('scaling...');             fig_scaling()
    print('done, figures in', FIG)
