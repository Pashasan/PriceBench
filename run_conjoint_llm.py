"""
Feed conjoint hotel-room tasks to an LLM via Ollama and record choices.

Supports both binary (A/B) and ternary (A/B/C) tasks from the new
10-attribute realistic hotel design.

Usage:
    python run_conjoint_llm.py --model qwen3:0.6b
    python run_conjoint_llm.py --model qwen3:0.6b --swap

Use --swap to present options in reversed order:
  - Binary: B shown as "Option A" and A shown as "Option B"
  - Ternary: C shown as "Option A", B stays "Option B", A shown as "Option C"
The recorded choice is flipped back to the original task frame.
Task IDs for swapped runs are offset by +10000 to distinguish them.

Outputs:
    conjoint_results_{model_tag}.csv       (original ordering)
    conjoint_results_{model_tag}_swap.csv  (reversed ordering)
"""

import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from ollama import Client

# ── Defaults ─────────────────────────────────────────────────────────────────
CONJOINT_FILE = 'conjoint_tasks.csv'
MAX_RETRIES = 3
TASK_ID_SWAP_OFFSET = 10000

# ── Prompt templates ─────────────────────────────────────────────────────────

BINARY_PROMPT_TEMPLATE = """\
You are booking a hotel room in New York City for a one-night stay. \
You must choose between the following two options.

Option A:
  Hotel: {a_name}
  Star rating: {a_stars} stars
  Neighborhood: {a_neighborhood}
  Guest review score: {a_review_score}/10 ({a_review_count:,} reviews)
  Room type: {a_room_type}
  Free cancellation: {a_cancellation_free}
  Breakfast included: {a_breakfast_included}
  Key amenities: {a_amenities}
  Price per night: ${a_price:,}

Option B:
  Hotel: {b_name}
  Star rating: {b_stars} stars
  Neighborhood: {b_neighborhood}
  Guest review score: {b_review_score}/10 ({b_review_count:,} reviews)
  Room type: {b_room_type}
  Free cancellation: {b_cancellation_free}
  Breakfast included: {b_breakfast_included}
  Key amenities: {b_amenities}
  Price per night: ${b_price:,}

Which option do you choose? Reply with only the letter A or B."""

TERNARY_PROMPT_TEMPLATE = """\
You are booking a hotel room in New York City for a one-night stay. \
You must choose one of the following three options.

Option A:
  Hotel: {a_name}
  Star rating: {a_stars} stars
  Neighborhood: {a_neighborhood}
  Guest review score: {a_review_score}/10 ({a_review_count:,} reviews)
  Room type: {a_room_type}
  Free cancellation: {a_cancellation_free}
  Breakfast included: {a_breakfast_included}
  Key amenities: {a_amenities}
  Price per night: ${a_price:,}

Option B:
  Hotel: {b_name}
  Star rating: {b_stars} stars
  Neighborhood: {b_neighborhood}
  Guest review score: {b_review_score}/10 ({b_review_count:,} reviews)
  Room type: {b_room_type}
  Free cancellation: {b_cancellation_free}
  Breakfast included: {b_breakfast_included}
  Key amenities: {b_amenities}
  Price per night: ${b_price:,}

Option C:
  Hotel: {c_name}
  Star rating: {c_stars} stars
  Neighborhood: {c_neighborhood}
  Guest review score: {c_review_score}/10 ({c_review_count:,} reviews)
  Room type: {c_room_type}
  Free cancellation: {c_cancellation_free}
  Breakfast included: {c_breakfast_included}
  Key amenities: {c_amenities}
  Price per night: ${c_price:,}

Which option do you choose? Reply with only the letter A, B, or C."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def yes_no(val):
    """Convert 0/1 to Yes/No for display."""
    return 'Yes' if int(val) == 1 else 'No'


def parse_choice(text, valid_choices=('A', 'B')):
    """Extract a valid choice letter from model output. Returns None on failure."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    valid_upper = tuple(c.upper() for c in valid_choices)
    # exact match first
    if text.upper() in valid_upper:
        return text.upper()
    # look for choice letter as standalone word (possibly preceded by "Option")
    pattern = r'\b(?:option\s+)?([' + ''.join(valid_upper) + r'])\b'
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # fallback: find any valid letter
    pattern = '[' + ''.join(valid_upper) + ']'
    m = re.search(pattern, text)
    return m.group(0) if m else None


def model_tag(name: str) -> str:
    return re.sub(r'[^\w\-]', '', name.replace(':', '_').replace('.', '_'))


def format_prompt(row, task_type, swap=False):
    """Build the prompt string for a task row."""
    def option_fields(prefix):
        return {
            f'{prefix}_name': row[f'{prefix}_name'],
            f'{prefix}_stars': int(row[f'{prefix}_stars']),
            f'{prefix}_neighborhood': row[f'{prefix}_neighborhood'],
            f'{prefix}_review_score': row[f'{prefix}_review_score'],
            f'{prefix}_review_count': int(row[f'{prefix}_review_count']),
            f'{prefix}_room_type': row[f'{prefix}_room_type'],
            f'{prefix}_cancellation_free': yes_no(row[f'{prefix}_cancellation_free']),
            f'{prefix}_breakfast_included': yes_no(row[f'{prefix}_breakfast_included']),
            f'{prefix}_amenities': row[f'{prefix}_amenities'],
            f'{prefix}_price': int(row[f'{prefix}_price']),
        }

    if task_type == 'binary':
        if swap:
            # Show B as Option A, A as Option B
            fields = {}
            for key, val in option_fields('b').items():
                fields[key.replace('b_', 'a_')] = val
            for key, val in option_fields('a').items():
                fields[key.replace('a_', 'b_')] = val
        else:
            fields = {**option_fields('a'), **option_fields('b')}
        return BINARY_PROMPT_TEMPLATE.format(**fields)
    else:
        # Ternary
        if swap:
            # Show C as A, B stays B, A becomes C
            fields = {}
            for key, val in option_fields('c').items():
                fields[key.replace('c_', 'a_')] = val
            fields.update(option_fields('b'))
            for key, val in option_fields('a').items():
                fields[key.replace('a_', 'c_')] = val
        else:
            fields = {
                **option_fields('a'),
                **option_fields('b'),
                **option_fields('c'),
            }
        return TERNARY_PROMPT_TEMPLATE.format(**fields)


# Flip maps for recording choices in original frame
BINARY_FLIP = {'A': 'B', 'B': 'A'}
TERNARY_FLIP = {'A': 'C', 'B': 'B', 'C': 'A'}  # C->A, B->B, A->C


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Run conjoint LLM evaluation')
    ap.add_argument('--model', required=True,
                    help='Ollama model name (e.g. qwen3:0.6b)')
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='Sampling temperature (0 = deterministic)')
    ap.add_argument('--top_p', type=float, default=1.0,
                    help='Nucleus-sampling p (irrelevant at temperature=0)')
    ap.add_argument('--top_k', type=int, default=1,
                    help='Top-k sampling (1 = greedy, irrelevant at temperature=0)')
    ap.add_argument('--seed', type=int, default=42,
                    help='RNG seed for the model (helps reproducibility)')
    ap.add_argument('--swap', action='store_true',
                    help='Present options in reversed order for counterbalancing')
    ap.add_argument('--no_think', action='store_true',
                    help='Disable thinking for reasoning models')
    ap.add_argument('--tag', type=str, default='',
                    help='Extra tag appended to output filename')
    ap.add_argument('--task_type', type=str, default='all',
                    choices=['all', 'binary', 'ternary'],
                    help='Which task types to run (default: all)')
    ap.add_argument('--tasks_file', default=CONJOINT_FILE)
    ap.add_argument('--checkpoint_every', type=int, default=10)
    args = ap.parse_args()

    tag = model_tag(args.model)
    suffix = '_swap' if args.swap else ''
    extra = f'_{args.tag}' if args.tag else ''
    results_file = f'conjoint_results_{tag}{extra}{suffix}.csv'
    client = Client()

    # ── Load tasks ────────────────────────────────────────────────────────
    tasks = pd.read_csv(args.tasks_file)

    # Filter by task type if requested
    if args.task_type != 'all':
        tasks = tasks[tasks['task_type'] == args.task_type].copy()

    # Offset task IDs for swapped runs so they don't collide
    if args.swap:
        tasks['task_id'] = tasks['task_id'] + TASK_ID_SWAP_OFFSET

    # ── Resume support ────────────────────────────────────────────────────
    done_ids = set()
    if os.path.exists(results_file):
        try:
            already = pd.read_csv(results_file)
            valid = already[already['choice'].isin(['A', 'B', 'C'])]
            done_ids = set(valid['task_id'].astype(int).tolist())
        except Exception:
            pass

    to_do = tasks[~tasks['task_id'].isin(done_ids)]

    if len(to_do) == 0:
        print(f"All {len(tasks)} tasks already scored in {results_file}")
        return

    swap_label = ' [SWAPPED]' if args.swap else ''
    n_binary = (to_do['task_type'] == 'binary').sum()
    n_ternary = (to_do['task_type'] == 'ternary').sum()
    print(f"Scoring {len(to_do)} tasks{swap_label}  "
          f"({n_binary} binary, {n_ternary} ternary)  "
          f"(model={args.model}, temp={args.temperature}, seed={args.seed})")

    # ── Ensure output file exists with header ─────────────────────────────
    if not os.path.exists(results_file):
        pd.DataFrame(columns=['task_id', 'choice', 'raw_output']).to_csv(
            results_file, index=False)

    buffer = []

    with tqdm(total=len(to_do), desc='Conjoint tasks',
              dynamic_ncols=True) as pbar:
        for _, row in to_do.iterrows():
            task_type = row['task_type']
            valid_choices = ('A', 'B', 'C') if task_type == 'ternary' \
                else ('A', 'B')

            prompt = format_prompt(row, task_type, swap=args.swap)

            choice = None
            raw = ''
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    retry_prompt = prompt if attempt == 1 else (
                        prompt + f"\n\nRespond with ONLY the letter "
                        f"{' or '.join(valid_choices)}.")
                    gen_kwargs = dict(
                        model=args.model,
                        prompt=retry_prompt,
                        options={
                            'temperature': args.temperature,
                            'top_p': args.top_p,
                            'top_k': args.top_k,
                            'seed': args.seed,
                            'num_predict': 16,
                        },
                        keep_alive='30m',
                    )
                    if args.no_think:
                        gen_kwargs['think'] = False
                    resp = client.generate(**gen_kwargs)
                    raw = (resp.get('response') or '').strip()
                    choice = parse_choice(raw, valid_choices)
                    if choice:
                        break
                    print(f"  [RETRY] task_id={row['task_id']} "
                          f"bad output (attempt {attempt}): {raw!r}")
                except Exception as e:
                    print(f"  [ERROR] task_id={row['task_id']} "
                          f"(attempt {attempt}): {e}")
                    raw = ''

            # Flip choice back to original frame
            if args.swap and choice:
                flip = TERNARY_FLIP if task_type == 'ternary' \
                    else BINARY_FLIP
                recorded_choice = flip.get(choice, choice)
            else:
                recorded_choice = choice

            buffer.append({
                'task_id': int(row['task_id']),
                'choice': recorded_choice,
                'raw_output': raw,
            })

            if len(buffer) >= args.checkpoint_every:
                pd.DataFrame(buffer).to_csv(
                    results_file, mode='a', header=False, index=False)
                buffer.clear()

            pbar.update(1)

    # flush remaining
    if buffer:
        pd.DataFrame(buffer).to_csv(
            results_file, mode='a', header=False, index=False)

    print(f"\nDone. Results saved to {results_file}")


if __name__ == '__main__':
    main()
