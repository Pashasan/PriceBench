"""
Shared configuration for all estimation and plotting scripts.

To add a new LLM, add one entry to MODELS below.  All scripts import from here
so nothing else needs to change.
"""

import numpy as np
import pandas as pd

# ── File paths ────────────────────────────────────────────────────────────────
TASKS_FILE = 'conjoint_tasks.csv'
TASK_ID_SWAP_OFFSET = 10000

# ── LLM registry ─────────────────────────────────────────────────────────────
# Keys   = display names used in tables and plot legends
# Values = (original_results_csv, swap_results_csv)
MODELS = {
    'Qwen3 0.6B': ('conjoint_results_qwen3_0_6b.csv',
                    'conjoint_results_qwen3_0_6b_swap.csv'),
    'Qwen3 1.7B': ('conjoint_results_qwen3_1_7b.csv',
                    'conjoint_results_qwen3_1_7b_swap.csv'),
    'Gemma2 9B': ('conjoint_results_gemma2_9b.csv',
                  'conjoint_results_gemma2_9b_swap.csv'),
    'Gemma3 1B': ('conjoint_results_gemma3_1b.csv',
                  'conjoint_results_gemma3_1b_swap.csv'),
    'Gemma3 4B': ('conjoint_results_gemma3_latest.csv',
                  'conjoint_results_gemma3_latest_swap.csv'),
    'Llama3.2 1B': ('conjoint_results_llama3_2_1b.csv',
                    'conjoint_results_llama3_2_1b_swap.csv'),
    'Llama3.2 3B': ('conjoint_results_llama3_2_latest.csv',
                    'conjoint_results_llama3_2_latest_swap.csv'),
    'Qwen3 4B': ('conjoint_results_qwen3_4b-instruct.csv',
                  'conjoint_results_qwen3_4b-instruct_swap.csv'),
    'Qwen3 8B': ('conjoint_results_qwen3_latest.csv',
                  'conjoint_results_qwen3_latest_swap.csv'),
    'Gemma3 12B': ('conjoint_results_gemma3_12b.csv',
                    'conjoint_results_gemma3_12b_swap.csv'),
    'Gemma3 27B': ('conjoint_results_gemma3_27b.csv',
                    'conjoint_results_gemma3_27b_swap.csv'),
    'Qwen3 30B-A3B': ('conjoint_results_qwen3_30b-a3b-instruct-2507-q4_K_M.csv',
                       'conjoint_results_qwen3_30b-a3b-instruct-2507-q4_K_M_swap.csv'),
    'Llama3 8B': ('conjoint_results_llama3_8b-instruct-fp16.csv',
                   'conjoint_results_llama3_8b-instruct-fp16_swap.csv'),
    'Llama3.1 8B': ('conjoint_results_llama3_1_8b.csv',
                     'conjoint_results_llama3_1_8b_swap.csv'),
    'DeepSeek-R1 1.5B': ('conjoint_results_deepseek-r1_1_5b.csv',
                          'conjoint_results_deepseek-r1_1_5b_swap.csv'),
    'DeepSeek-R1 7B': ('conjoint_results_deepseek-r1_7b.csv',
                        'conjoint_results_deepseek-r1_7b_swap.csv'),
    'Mistral 7B': ('conjoint_results_mistral_7b-instruct-fp16.csv',
                    'conjoint_results_mistral_7b-instruct-fp16_swap.csv'),
    'Mistral-Nemo 12B': ('conjoint_results_mistral-nemo.csv',
                          'conjoint_results_mistral-nemo_swap.csv'),
    'Phi-2 2.7B': ('conjoint_results_phi.csv',
                    'conjoint_results_phi_swap.csv'),
    'Phi-3 Mini': ('conjoint_results_phi3_mini.csv',
                    'conjoint_results_phi3_mini_swap.csv'),
    'Phi-3 Medium': ('conjoint_results_phi3_medium.csv',
                      'conjoint_results_phi3_medium_swap.csv'),
    'Phi-4 Mini': ('conjoint_results_phi4-mini.csv',
                    'conjoint_results_phi4-mini_swap.csv'),
    'Phi-4 14B': ('conjoint_results_phi4_14b.csv',
                   'conjoint_results_phi4_14b_swap.csv'),
    'GPT-4.1 Nano': ('conjoint_results_gpt-4_1-nano.csv',
                      'conjoint_results_gpt-4_1-nano_swap.csv'),
    'GPT-4.1 Mini': ('conjoint_results_gpt-4_1-mini.csv',
                      'conjoint_results_gpt-4_1-mini_swap.csv'),
    'GPT-5.4 Nano': ('conjoint_results_gpt-5_4-nano.csv',
                      'conjoint_results_gpt-5_4-nano_swap.csv'),
    'GPT-5.4 Mini': ('conjoint_results_gpt-5_4-mini.csv',
                      'conjoint_results_gpt-5_4-mini_swap.csv'),
    'GPT-5.4': ('conjoint_results_gpt-5_4.csv',
                 'conjoint_results_gpt-5_4_swap.csv'),
    # Claude (Anthropic API)
    'Claude Haiku 4.5': ('conjoint_results_claude-haiku-4-5.csv',
                          'conjoint_results_claude-haiku-4-5_swap.csv'),
}

# ── Attribute encoding ────────────────────────────────────────────────────────
# Room types are diverse (69 unique); we encode by bed size extracted from name
BED_SIZE_RANK = {
    'single': 0, 'twin': 0,
    'double': 1, 'full': 1, 'bunk': 1,
    'queen': 2,
    'king': 3,
}

N_DECILES = 10


def extract_bed_size(room_type):
    """Extract bed size category from room type string."""
    rt = room_type.lower()
    for key in ['king', 'queen', 'double', 'full', 'twin', 'single', 'bunk']:
        if key in rt:
            return BED_SIZE_RANK[key]
    return 1  # default to double if unknown


# ── Neighborhood area groupings ──────────────────────────────────────────────
# 36 neighborhoods -> 7 areas (for area controls when not using pair FE)
AREA_MAP = {
    'Midtown': 'Midtown',
    'Midtown East': 'Midtown',
    'Midtown West': 'Midtown',
    'Upper Midtown': 'Midtown',
    'Times Square': 'Midtown',
    'Theater District': 'Midtown',
    "Hell's Kitchen": 'Midtown',
    'Hudson Yards': 'Midtown',
    'Financial District': 'Downtown',
    'TriBeCa': 'Downtown',
    'SoHo': 'Downtown',
    'Little Italy': 'Downtown',
    'NoHo': 'Downtown',
    'Lower East Side': 'Downtown',
    'Chelsea': 'Chelsea/Flatiron/NoMad',
    'Flatiron': 'Chelsea/Flatiron/NoMad',
    'NoMad': 'Chelsea/Flatiron/NoMad',
    'Meatpacking District': 'Chelsea/Flatiron/NoMad',
    'Gramercy': 'Villages',
    'Union Square': 'Villages',
    'Greenwich Village': 'Villages',
    'West Village': 'Villages',
    'East Village': 'Villages',
    'Murray Hill': 'Villages',
    'Upper East Side': 'Upper Manhattan',
    'Upper West Side': 'Upper Manhattan',
    'Washington Heights': 'Upper Manhattan',
    'Brooklyn': 'Brooklyn',
    'Brooklyn Heights': 'Brooklyn',
    'Downtown Brooklyn': 'Brooklyn',
    'Williamsburg, Brooklyn': 'Brooklyn',
    'Boerum Hill, Brooklyn': 'Brooklyn',
    'Greenpoint, Brooklyn': 'Brooklyn',
    'Park Slope, Brooklyn': 'Brooklyn',
    'Long Island City, Queens': 'Queens',
    'Rockaway Beach, Queens': 'Queens',
}


# ── Shared data-loading helpers ───────────────────────────────────────────────

def load_choices(model_name):
    """Concatenate original + swap results for one LLM, aligned to task_ids."""
    orig_file, swap_file = MODELS[model_name]
    orig = pd.read_csv(orig_file)
    swap = pd.read_csv(swap_file)
    swap['task_id'] = swap['task_id'] - TASK_ID_SWAP_OFFSET
    return pd.concat([orig, swap], ignore_index=True)


def build_dataset(model_name, task_type='binary', decile_edges=None):
    """Build the analysis dataset for one LLM.

    Parameters
    ----------
    model_name : str
        Key in MODELS dict.
    task_type : str
        'binary' or 'ternary'. For binary, returns difference variables.
        For ternary, returns long-format data.
    decile_edges : array-like, optional
        If provided, adds difference-coded decile dummies (dd_D2..dd_D10).
        Only used for binary task_type.

    Returns
    -------
    pd.DataFrame
    """
    tasks = pd.read_csv(TASKS_FILE)
    tasks = tasks[tasks['task_type'] == task_type].copy()
    results = load_choices(model_name)
    df = results.merge(tasks, on='task_id', how='inner')

    if task_type == 'binary':
        return _build_binary(df, decile_edges)
    else:
        return _build_ternary(df)


def _build_binary(df, decile_edges=None):
    """Build difference-coded dataset for binary choice logit."""
    df['choice_A'] = (df['choice'] == 'A').astype(int)

    # Price differences
    df['d_price'] = df['a_price'] - df['b_price']
    df['d_log_price'] = np.log(df['a_price']) - np.log(df['b_price'])

    # Control variable differences (randomized attributes only)
    df['a_bed_rank'] = df['a_room_type'].apply(extract_bed_size)
    df['b_bed_rank'] = df['b_room_type'].apply(extract_bed_size)
    df['d_bed'] = df['a_bed_rank'] - df['b_bed_rank']
    df['d_cancel'] = df['a_cancellation_free'] - df['b_cancellation_free']
    df['d_breakfast'] = df['a_breakfast_included'] - df['b_breakfast_included']
    df['d_review_score'] = df['a_review_score'] - df['b_review_score']
    df['d_review_count_100'] = (
        (df['a_review_count'] - df['b_review_count']) / 100
    )

    # Fixed attribute differences (for non-pair-FE specifications)
    df['d_stars'] = df['a_stars'] - df['b_stars']
    df['a_area'] = df['a_neighborhood'].map(AREA_MAP).fillna('Other')
    df['b_area'] = df['b_neighborhood'].map(AREA_MAP).fillna('Other')

    # Decile dummies
    if decile_edges is not None:
        labels = [f'D{i}' for i in range(1, N_DECILES + 1)]
        df['a_decile'] = pd.cut(
            df['a_price'], bins=decile_edges, labels=labels
        )
        df['b_decile'] = pd.cut(
            df['b_price'], bins=decile_edges, labels=labels
        )
        for i in range(2, N_DECILES + 1):
            d = f'D{i}'
            df[f'dd_{d}'] = (
                (df['a_decile'] == d).astype(int)
                - (df['b_decile'] == d).astype(int)
            )

    return df


def _build_ternary(df):
    """Build long-format dataset for multinomial conditional logit."""
    rows = []
    for _, task in df.iterrows():
        choice = task['choice']
        for opt in ['a', 'b', 'c']:
            rows.append({
                'task_id': task['task_id'],
                'group_id': task['group_id'],
                'option': opt.upper(),
                'chosen': int(choice == opt.upper()),
                'price': task[f'{opt}_price'],
                'log_price': np.log(task[f'{opt}_price']),
                'stars': task[f'{opt}_stars'],
                'bed_rank': extract_bed_size(task[f'{opt}_room_type']),
                'cancel': int(task[f'{opt}_cancellation_free']),
                'breakfast': int(task[f'{opt}_breakfast_included']),
                'review_score': task[f'{opt}_review_score'],
                'review_count_100': task[f'{opt}_review_count'] / 100,
                'neighborhood': task[f'{opt}_neighborhood'],
                'area': AREA_MAP.get(task[f'{opt}_neighborhood'], 'Other'),
            })
    return pd.DataFrame(rows)


def compute_decile_edges(task_type='binary'):
    """Compute decile bin edges from pooled prices in conjoint_tasks.csv."""
    tasks = pd.read_csv(TASKS_FILE)
    tasks = tasks[tasks['task_type'] == task_type]
    cols = ['a_price', 'b_price']
    if task_type == 'ternary':
        cols.append('c_price')
    all_prices = pd.concat([tasks[c] for c in cols])
    edges = np.percentile(all_prices, np.arange(0, 110, 10))
    edges[0] -= 0.5
    return edges


def decile_medians(edges, task_type='binary'):
    """Return median price within each decile."""
    tasks = pd.read_csv(TASKS_FILE)
    tasks = tasks[tasks['task_type'] == task_type]
    cols = ['a_price', 'b_price']
    if task_type == 'ternary':
        cols.append('c_price')
    all_prices = pd.concat([tasks[c] for c in cols])
    labels = [f'D{i}' for i in range(1, N_DECILES + 1)]
    assigned = pd.cut(all_prices, bins=edges, labels=labels)
    return all_prices.groupby(assigned, observed=False).median().values


def decile_price_ranges(edges, task_type='binary'):
    """Return (min_array, max_array) for each decile bin."""
    tasks = pd.read_csv(TASKS_FILE)
    tasks = tasks[tasks['task_type'] == task_type]
    cols = ['a_price', 'b_price']
    if task_type == 'ternary':
        cols.append('c_price')
    all_prices = pd.concat([tasks[c] for c in cols])
    labels = [f'D{i}' for i in range(1, N_DECILES + 1)]
    assigned = pd.cut(all_prices, bins=edges, labels=labels)
    grp = all_prices.groupby(assigned, observed=False)
    return grp.min().values, grp.max().values


def fmt_price(v):
    """Format a price for axis labels ($123 or $1.5k)."""
    if v >= 1000:
        return f'${v/1000:.1f}k'
    return f'${v:.0f}'


def decile_tick_labels(edges, task_type='binary'):
    """Return x-tick labels like '$20-$37' for each decile."""
    d_min, d_max = decile_price_ranges(edges, task_type)
    return [f'{fmt_price(lo)}-{fmt_price(hi)}' for lo, hi in zip(d_min, d_max)]


def sig_stars(p):
    """Significance stars for p-values."""
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''
