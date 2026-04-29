"""
Feed conjoint hotel-room tasks to an LLM via OpenAI-compatible API and record choices.

Supports both binary (A/B) and ternary (A/B/C) tasks.
Works with OpenAI, Anthropic (via OpenAI-compat), and other providers.

Usage:
    python run_conjoint_llm_api.py --model gpt-4.1-nano --api_key sk-...
    python run_conjoint_llm_api.py --model gpt-4.1-nano --api_key sk-... --swap

Use --swap to present options in reversed order (same logic as run_conjoint_llm.py).
Task IDs for swapped runs are offset by +10000.

Outputs:
    conjoint_results_{model_tag}.csv       (original ordering)
    conjoint_results_{model_tag}_swap.csv  (reversed ordering)
"""

import os
import re
import time
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ── Defaults ─────────────────────────────────────────────────────────────────
CONJOINT_FILE = 'conjoint_tasks.csv'
MAX_RETRIES = 3
TASK_ID_SWAP_OFFSET = 10000
RETRY_DELAY = 2  # seconds between retries on API error

# ── Prompt templates (identical to run_conjoint_llm.py) ─────────────────────

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


# ── Helpers (same as run_conjoint_llm.py) ───────────────────────────────────

def yes_no(val):
    return 'Yes' if int(val) == 1 else 'No'


def parse_choice(text, valid_choices=('A', 'B')):
    if not isinstance(text, str):
        return None
    text = text.strip()
    valid_upper = tuple(c.upper() for c in valid_choices)
    if text.upper() in valid_upper:
        return text.upper()
    pattern = r'\b(?:option\s+)?([' + ''.join(valid_upper) + r'])\b'
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    pattern = '[' + ''.join(valid_upper) + ']'
    m = re.search(pattern, text)
    return m.group(0) if m else None


def model_tag(name: str) -> str:
    return re.sub(r'[^\w\-]', '', name.replace(':', '_').replace('.', '_').replace('/', '_'))


def format_prompt(row, task_type, swap=False):
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
            fields = {}
            for key, val in option_fields('b').items():
                fields[key.replace('b_', 'a_')] = val
            for key, val in option_fields('a').items():
                fields[key.replace('a_', 'b_')] = val
        else:
            fields = {**option_fields('a'), **option_fields('b')}
        return BINARY_PROMPT_TEMPLATE.format(**fields)
    else:
        if swap:
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


BINARY_FLIP = {'A': 'B', 'B': 'A'}
TERNARY_FLIP = {'A': 'C', 'B': 'B', 'C': 'A'}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Run conjoint LLM evaluation via API')
    ap.add_argument('--model', required=True,
                    help='API model name (e.g. gpt-4.1-nano)')
    ap.add_argument('--api_key', type=str, default=None,
                    help='API key (or set OPENAI_API_KEY env var)')
    ap.add_argument('--base_url', type=str, default=None,
                    help='Custom API base URL (for non-OpenAI providers)')
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='Sampling temperature (0 = deterministic)')
    ap.add_argument('--seed', type=int, default=42,
                    help='RNG seed (if supported by provider)')
    ap.add_argument('--swap', action='store_true',
                    help='Present options in reversed order')
    ap.add_argument('--tag', type=str, default='',
                    help='Extra tag appended to output filename')
    ap.add_argument('--task_type', type=str, default='all',
                    choices=['all', 'binary', 'ternary'],
                    help='Which task types to run (default: all)')
    ap.add_argument('--tasks_file', default=CONJOINT_FILE)
    ap.add_argument('--checkpoint_every', type=int, default=10)
    ap.add_argument('--max_tokens', type=int, default=16,
                    help='Max tokens in response')
    args = ap.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: Provide --api_key or set OPENAI_API_KEY env var")
        return

    # Build client
    client_kwargs = {'api_key': api_key}
    if args.base_url:
        client_kwargs['base_url'] = args.base_url
    client = OpenAI(**client_kwargs)

    tag = model_tag(args.model)
    suffix = '_swap' if args.swap else ''
    extra = f'_{args.tag}' if args.tag else ''
    results_file = f'conjoint_results_{tag}{extra}{suffix}.csv'

    # ── Load tasks ────────────────────────────────────────────────────────
    tasks = pd.read_csv(args.tasks_file)
    if args.task_type != 'all':
        tasks = tasks[tasks['task_type'] == args.task_type].copy()
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

                    api_kwargs = dict(
                        model=args.model,
                        messages=[
                            {"role": "user", "content": retry_prompt}
                        ],
                        temperature=args.temperature,
                        seed=args.seed,
                    )
                    # Newer OpenAI models (gpt-5.*) require
                    # max_completion_tokens; older ones use max_tokens.
                    if args.model.startswith(('gpt-5', 'o3', 'o4')):
                        api_kwargs['max_completion_tokens'] = args.max_tokens
                    else:
                        api_kwargs['max_tokens'] = args.max_tokens
                    resp = client.chat.completions.create(**api_kwargs)
                    raw = (resp.choices[0].message.content or '').strip()
                    choice = parse_choice(raw, valid_choices)
                    if choice:
                        break
                    print(f"  [RETRY] task_id={row['task_id']} "
                          f"bad output (attempt {attempt}): {raw!r}")
                except Exception as e:
                    err_str = str(e)
                    print(f"  [ERROR] task_id={row['task_id']} "
                          f"(attempt {attempt}): {err_str}")
                    raw = ''
                    # Back off on rate limits
                    if 'rate' in err_str.lower() or '429' in err_str:
                        wait = RETRY_DELAY * attempt * 5
                        print(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        time.sleep(RETRY_DELAY * attempt)

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
