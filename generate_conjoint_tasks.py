"""
Generate 3,600 realistic conjoint choice tasks for NYC hotel rooms.

Design (per COMPREHENSIVE_PLAN.md):
  - Pool: 179 realistic NYC hotel profiles from hotel_pool.json
  - Binary tasks: 450 unique hotel pairs x 4 repetitions = 1,800 tasks
  - Ternary tasks: 300 unique hotel triples x 6 repetitions = 1,800 tasks
  - Total: 3,600 tasks

Each repetition draws fresh random values for variable attributes:
  - Price: uniform integer from [price_min, price_max]
  - Room type: uniform from hotel's room_types list
  - Free cancellation: Bernoulli(cancellation_free_prob)
  - Breakfast included: Bernoulli(breakfast_prob)
  - Review score: base +/- 0.2 (uniform)
  - Review count: base +/- 10% (uniform, integer)

Fixed attributes (name, stars, neighborhood, amenities) are constant
across all appearances of the same hotel.

Position counterbalancing is handled at scoring time (--swap flag in
run_conjoint_llm.py), not in the task design.

Usage:
    python generate_conjoint_tasks.py [--seed 2026]
"""

import json
import argparse
import itertools
import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
HOTEL_POOL_FILE = 'hotel_pool.json'
OUTPUT_FILE = 'conjoint_tasks.csv'

N_BINARY_PAIRS = 450
N_BINARY_REPS = 4
N_TERNARY_TRIPLES = 300
N_TERNARY_REPS = 6

N_BINARY_TASKS = N_BINARY_PAIRS * N_BINARY_REPS      # 1,800
N_TERNARY_TASKS = N_TERNARY_TRIPLES * N_TERNARY_REPS  # 1,800
N_TOTAL = N_BINARY_TASKS + N_TERNARY_TASKS             # 3,600

DEFAULT_SEED = 2026


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_hotel_pool(path=HOTEL_POOL_FILE):
    """Load hotel profiles from JSON."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data['hotels']


def draw_variable_attributes(hotel, rng):
    """Draw fresh random values for a hotel's variable attributes.

    Returns a dict with all 11 fields needed per option in the task CSV.
    """
    price = int(rng.integers(hotel['price_min'], hotel['price_max'] + 1))
    room_type = str(rng.choice(hotel['room_types']))
    cancellation_free = int(rng.random() < hotel['cancellation_free_prob'])
    breakfast_included = int(rng.random() < hotel['breakfast_prob'])
    review_score = round(
        hotel['review_score_base'] + float(rng.uniform(-0.2, 0.2)), 1
    )
    review_count = int(round(
        hotel['review_count_base'] * (1 + float(rng.uniform(-0.1, 0.1)))
    ))

    return {
        'hotel_id': hotel['id'],
        'name': hotel['name'],
        'stars': hotel['stars'],
        'neighborhood': hotel['neighborhood'],
        'review_score': review_score,
        'review_count': review_count,
        'room_type': room_type,
        'cancellation_free': cancellation_free,
        'breakfast_included': breakfast_included,
        'amenities': '; '.join(hotel['amenities']),
        'price': price,
    }


OPTION_KEYS = [
    'hotel_id', 'name', 'stars', 'neighborhood',
    'review_score', 'review_count', 'room_type',
    'cancellation_free', 'breakfast_included', 'amenities', 'price',
]


def generate_tasks(hotels, seed=DEFAULT_SEED):
    """Generate the full set of 3,600 conjoint tasks."""
    rng = np.random.default_rng(seed)
    n_hotels = len(hotels)

    tasks = []
    task_id = 1

    # ── Sample 450 unique pairs ──────────────────────────────────────────
    # C(179, 2) = 15,931 possible pairs; we sample 450
    all_pairs = list(itertools.combinations(range(n_hotels), 2))
    selected_pair_idx = rng.choice(
        len(all_pairs), size=N_BINARY_PAIRS, replace=False
    )
    sampled_pairs = [all_pairs[i] for i in sorted(selected_pair_idx)]

    # ── Binary tasks: 450 pairs x 4 reps = 1,800 ────────────────────────
    for group_id, (i, j) in enumerate(sampled_pairs, start=1):
        hotel_a = hotels[i]
        hotel_b = hotels[j]

        for rep in range(1, N_BINARY_REPS + 1):
            attrs_a = draw_variable_attributes(hotel_a, rng)
            attrs_b = draw_variable_attributes(hotel_b, rng)

            row = {
                'task_id': task_id,
                'task_type': 'binary',
                'group_id': group_id,
                'rep': rep,
            }
            for key in OPTION_KEYS:
                row[f'a_{key}'] = attrs_a[key]
            for key in OPTION_KEYS:
                row[f'b_{key}'] = attrs_b[key]

            tasks.append(row)
            task_id += 1

    # ── Sample 300 unique triples (rejection sampling) ───────────────────
    # C(179, 3) = 945,679 possible; rejection sampling is efficient
    sampled_triples = []
    seen_triples = set()
    while len(sampled_triples) < N_TERNARY_TRIPLES:
        triple = tuple(sorted(
            rng.choice(n_hotels, size=3, replace=False).tolist()
        ))
        if triple not in seen_triples:
            seen_triples.add(triple)
            sampled_triples.append(triple)

    # ── Ternary tasks: 300 triples x 6 reps = 1,800 ─────────────────────
    group_id_offset = N_BINARY_PAIRS  # binary groups are 1-450

    for group_idx, (i, j, k) in enumerate(sampled_triples, start=1):
        group_id = group_id_offset + group_idx  # 451-750
        hotel_a = hotels[i]
        hotel_b = hotels[j]
        hotel_c = hotels[k]

        for rep in range(1, N_TERNARY_REPS + 1):
            attrs_a = draw_variable_attributes(hotel_a, rng)
            attrs_b = draw_variable_attributes(hotel_b, rng)
            attrs_c = draw_variable_attributes(hotel_c, rng)

            row = {
                'task_id': task_id,
                'task_type': 'ternary',
                'group_id': group_id,
                'rep': rep,
            }
            for key in OPTION_KEYS:
                row[f'a_{key}'] = attrs_a[key]
            for key in OPTION_KEYS:
                row[f'b_{key}'] = attrs_b[key]
            for key in OPTION_KEYS:
                row[f'c_{key}'] = attrs_c[key]

            tasks.append(row)
            task_id += 1

    return pd.DataFrame(tasks)


# ── Validation ───────────────────────────────────────────────────────────────

def validate_dataset(df, hotels):
    """Run comprehensive validation checks. Raises AssertionError on failure."""
    hotel_by_id = {h['id']: h for h in hotels}
    errors = []

    # 1. Correct total counts
    binary = df[df['task_type'] == 'binary']
    ternary = df[df['task_type'] == 'ternary']

    if len(df) != N_TOTAL:
        errors.append(f'Total tasks: expected {N_TOTAL}, got {len(df)}')
    if len(binary) != N_BINARY_TASKS:
        errors.append(f'Binary tasks: expected {N_BINARY_TASKS}, got {len(binary)}')
    if len(ternary) != N_TERNARY_TASKS:
        errors.append(f'Ternary tasks: expected {N_TERNARY_TASKS}, got {len(ternary)}')

    # 2. Unique task_ids
    if df['task_id'].nunique() != len(df):
        errors.append('Duplicate task_ids found')

    # 3. Correct group counts
    n_binary_groups = binary['group_id'].nunique()
    n_ternary_groups = ternary['group_id'].nunique()
    if n_binary_groups != N_BINARY_PAIRS:
        errors.append(f'Binary groups: expected {N_BINARY_PAIRS}, got {n_binary_groups}')
    if n_ternary_groups != N_TERNARY_TRIPLES:
        errors.append(f'Ternary groups: expected {N_TERNARY_TRIPLES}, got {n_ternary_groups}')

    # 4. Correct reps per group
    for gid, grp in binary.groupby('group_id'):
        if len(grp) != N_BINARY_REPS:
            errors.append(f'Binary group {gid}: expected {N_BINARY_REPS} reps, got {len(grp)}')
    for gid, grp in ternary.groupby('group_id'):
        if len(grp) != N_TERNARY_REPS:
            errors.append(f'Ternary group {gid}: expected {N_TERNARY_REPS} reps, got {len(grp)}')

    # 5. Fixed attributes constant within each group
    fixed_attrs = ['hotel_id', 'name', 'stars', 'neighborhood', 'amenities']
    for prefix in ['a', 'b']:
        for attr in fixed_attrs:
            col = f'{prefix}_{attr}'
            for gid, grp in df.groupby('group_id'):
                if grp[col].nunique() != 1:
                    errors.append(
                        f'Fixed attr {col} varies within group {gid}'
                    )
    # c_* fixed attrs for ternary only
    for attr in fixed_attrs:
        col = f'c_{attr}'
        for gid, grp in ternary.groupby('group_id'):
            if grp[col].nunique() != 1:
                errors.append(f'Fixed attr {col} varies within ternary group {gid}')

    # 6. Prices within hotel bounds
    for prefix in ['a', 'b']:
        for _, row in df.iterrows():
            hid = int(row[f'{prefix}_hotel_id'])
            hotel = hotel_by_id[hid]
            price = row[f'{prefix}_price']
            if not (hotel['price_min'] <= price <= hotel['price_max']):
                errors.append(
                    f"Task {row['task_id']} {prefix}_price={price} "
                    f"outside [{hotel['price_min']}, {hotel['price_max']}] "
                    f"for hotel {hotel['name']}"
                )
    for _, row in ternary.iterrows():
        hid = int(row['c_hotel_id'])
        hotel = hotel_by_id[hid]
        price = row['c_price']
        if not (hotel['price_min'] <= price <= hotel['price_max']):
            errors.append(
                f"Task {row['task_id']} c_price={price} "
                f"outside [{hotel['price_min']}, {hotel['price_max']}] "
                f"for hotel {hotel['name']}"
            )

    # 7. Review scores within base +/- 0.2
    for prefix in ['a', 'b']:
        for _, row in df.iterrows():
            hid = int(row[f'{prefix}_hotel_id'])
            hotel = hotel_by_id[hid]
            score = row[f'{prefix}_review_score']
            expected_min = round(hotel['review_score_base'] - 0.2, 1)
            expected_max = round(hotel['review_score_base'] + 0.2, 1)
            if not (expected_min - 0.01 <= score <= expected_max + 0.01):
                errors.append(
                    f"Task {row['task_id']} {prefix}_review_score={score} "
                    f"outside [{expected_min}, {expected_max}] "
                    f"for hotel {hotel['name']}"
                )

    # 8. No pair appears in both binary and ternary sets as a sub-pair
    # (not strictly required but good to check)

    # 9. Variable attributes actually vary across reps of same group
    n_constant_price = 0
    for gid, grp in df.groupby('group_id'):
        for prefix in ['a', 'b']:
            if grp[f'{prefix}_price'].nunique() == 1 and len(grp) > 1:
                n_constant_price += 1
    if n_constant_price > 0:
        # Not an error per se (could happen by chance for narrow ranges)
        # but flag if it's excessive
        pct = n_constant_price / (df['group_id'].nunique() * 2) * 100
        if pct > 5:
            errors.append(
                f'{n_constant_price} option-groups have constant price '
                f'across all reps ({pct:.1f}%)'
            )

    # 10. Binary tasks should NOT have c_* columns populated
    if 'c_hotel_id' in binary.columns:
        non_null = binary['c_hotel_id'].notna().sum()
        if non_null > 0:
            errors.append(f'Binary tasks have {non_null} non-null c_hotel_id values')

    return errors


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Generate conjoint tasks from hotel pool'
    )
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--pool', default=HOTEL_POOL_FILE)
    ap.add_argument('--output', default=OUTPUT_FILE)
    args = ap.parse_args()

    hotels = load_hotel_pool(args.pool)
    print(f'Loaded {len(hotels)} hotel profiles from {args.pool}')

    df = generate_tasks(hotels, seed=args.seed)

    # ── Validation ────────────────────────────────────────────────────────
    print('\nValidating dataset...')
    errors = validate_dataset(df, hotels)
    if errors:
        print(f'\nVALIDATION FAILED ({len(errors)} errors):')
        for e in errors[:20]:
            print(f'  - {e}')
        if len(errors) > 20:
            print(f'  ... and {len(errors) - 20} more')
        raise SystemExit(1)
    print('All validation checks passed.')

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(args.output, index=False, encoding='utf-8')

    # ── Summary ───────────────────────────────────────────────────────────
    binary = df[df['task_type'] == 'binary']
    ternary = df[df['task_type'] == 'ternary']

    print(f'\nGenerated {len(df)} conjoint tasks -> {args.output}  (seed={args.seed})')
    print(f'  Binary:  {len(binary)} tasks '
          f'({binary["group_id"].nunique()} unique pairs x {N_BINARY_REPS} reps)')
    print(f'  Ternary: {len(ternary)} tasks '
          f'({ternary["group_id"].nunique()} unique triples x {N_TERNARY_REPS} reps)')
    print(f'  Task IDs: {df["task_id"].min()} - {df["task_id"].max()}')

    # Price summary
    all_prices = pd.concat([df['a_price'], df['b_price']])
    if 'c_price' in df.columns:
        c_prices = pd.to_numeric(ternary['c_price'], errors='coerce').dropna()
        all_prices = pd.concat([all_prices, c_prices])
    print(f'\nPrice summary (all options, all tasks):')
    print(all_prices.describe().to_string())

    # Hotel coverage
    hotel_ids_used = set()
    for prefix in ['a', 'b']:
        hotel_ids_used.update(df[f'{prefix}_hotel_id'].astype(int).tolist())
    if 'c_hotel_id' in ternary.columns:
        hotel_ids_used.update(
            ternary['c_hotel_id'].astype(int).tolist()
        )
    print(f'\nUnique hotels used: {len(hotel_ids_used)} / {len(hotels)}')

    # Star distribution across task appearances
    all_stars = pd.concat([df['a_stars'], df['b_stars']])
    if 'c_stars' in ternary.columns:
        all_stars = pd.concat([all_stars, ternary['c_stars']])
    star_counts = all_stars.value_counts().sort_index()
    print(f'\nStar distribution (option appearances):')
    for star, count in star_counts.items():
        print(f'  {star}-star: {count}')

    # Within-pair price variation
    print(f'\nWithin-group price variation (a_price std across reps):')
    for task_type, sub in [('binary', binary), ('ternary', ternary)]:
        stds = sub.groupby('group_id')['a_price'].std()
        print(f'  {task_type}: mean std=${stds.mean():.1f}, '
              f'median=${stds.median():.1f}, '
              f'min=${stds.min():.1f}, max=${stds.max():.1f}')


if __name__ == '__main__':
    main()
