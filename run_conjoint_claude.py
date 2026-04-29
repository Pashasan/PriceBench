"""
Feed conjoint hotel-room tasks to Claude models via Anthropic API.

Reuses prompt templates and parsing from run_conjoint_llm.py but calls the
Anthropic Messages API instead of Ollama.

Usage:
    python run_conjoint_claude.py --model claude-haiku-4-5
    python run_conjoint_claude.py --model claude-haiku-4-5 --swap

Set ANTHROPIC_API_KEY env var or pass --api_key.
"""

import os
import re
import argparse
import time
import pandas as pd
from tqdm import tqdm
import anthropic

from run_conjoint_llm import (
    CONJOINT_FILE, MAX_RETRIES, TASK_ID_SWAP_OFFSET,
    BINARY_FLIP, TERNARY_FLIP,
    parse_choice, format_prompt,
)

def model_tag(name: str) -> str:
    return re.sub(r'[^\w\-]', '', name.replace(':', '_').replace('.', '_'))


def main():
    ap = argparse.ArgumentParser(description='Run conjoint via Anthropic API')
    ap.add_argument('--model', required=True,
                    help='Anthropic model ID (e.g. claude-haiku-4-5)')
    ap.add_argument('--temperature', type=float, default=0.0,
                    help='Sampling temperature (0 = deterministic)')
    ap.add_argument('--swap', action='store_true',
                    help='Present options in reversed order')
    ap.add_argument('--tag', type=str, default='',
                    help='Extra tag appended to output filename')
    ap.add_argument('--task_type', type=str, default='all',
                    choices=['all', 'binary', 'ternary'],
                    help='Which task types to run (default: all)')
    ap.add_argument('--tasks_file', default=CONJOINT_FILE)
    ap.add_argument('--checkpoint_every', type=int, default=10)
    ap.add_argument('--api_key', type=str, default=None,
                    help='Anthropic API key (or set ANTHROPIC_API_KEY)')
    args = ap.parse_args()

    tag = model_tag(args.model)
    suffix = '_swap' if args.swap else ''
    extra = f'_{args.tag}' if args.tag else ''
    results_file = f'conjoint_results_{tag}{extra}{suffix}.csv'

    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError('Set ANTHROPIC_API_KEY or pass --api_key')
    client = anthropic.Anthropic(api_key=api_key)

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
          f"(model={args.model}, temp={args.temperature})")

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

                    resp = client.messages.create(
                        model=args.model,
                        max_tokens=16,
                        temperature=args.temperature,
                        messages=[{'role': 'user', 'content': retry_prompt}],
                    )
                    raw = resp.content[0].text.strip()
                    choice = parse_choice(raw, valid_choices)
                    if choice:
                        break
                    print(f"  [RETRY] task_id={row['task_id']} "
                          f"bad output (attempt {attempt}): {raw!r}")
                except anthropic.RateLimitError:
                    wait = 2 ** attempt
                    print(f"  [RATE LIMIT] waiting {wait}s...")
                    time.sleep(wait)
                except Exception as e:
                    print(f"  [ERROR] task_id={row['task_id']} "
                          f"(attempt {attempt}): {e}")
                    raw = ''

            if args.swap and choice:
                flip = TERNARY_FLIP if task_type == 'ternary' else BINARY_FLIP
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

    if buffer:
        pd.DataFrame(buffer).to_csv(
            results_file, mode='a', header=False, index=False)

    print(f"\nDone. Results saved to {results_file}")


if __name__ == '__main__':
    main()
