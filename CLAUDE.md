# Are LLMs Price Sensitive?

Conjoint choice experiment testing how 29 LLMs trade off price against quality
when choosing hotel rooms. The public-facing writeup lives in `web/`.

## Environment

- **Python**: Use the `datasci` conda environment.
  Executable: `/c/Users/pak_o/anaconda3/envs/datasci/python.exe`
  (conda cannot be activated directly in this shell; call the executable by path)
- **Key packages**: numpy, pandas, statsmodels, matplotlib, scipy, ollama, openai, anthropic, tqdm
- **Platform**: Windows (MINGW64/Git Bash)
- **Ollama**: Local LLM inference; must be running for `run_conjoint_llm.py`

## Project structure

### Shared configuration
- `config.py` -- single source of truth for the LLM registry (`MODELS` dict),
  attribute encoding (`BED_SIZE_RANK`), area mapping, and dataset builders. All
  analysis scripts import from here. To add a new model, only edit this file.

### Data generation
- `build_hotel_pool.py` -- constructs `hotel_pool.json` (179 NYC profiles)
- `generate_conjoint_tasks.py` -- creates `conjoint_tasks.csv` (3,600 tasks:
  1,800 binary + 1,800 ternary). Seed 2026.

### Scoring
- `run_conjoint_llm.py` -- Ollama scorer (open-weight models)
- `run_conjoint_llm_api.py` -- OpenAI-compatible API scorer
- `run_conjoint_claude.py` -- Anthropic SDK scorer for Claude

Each runner produces a pair of CSVs:
- `conjoint_results_{tag}.csv` (original ordering)
- `conjoint_results_{tag}_swap.csv` (reversed ordering, task IDs offset by +10000)

These are gitignored; share via Dropbox/Drive when needed.

### Analysis pipeline (`analysis/`)
- `analysis/triage.py` -- first-option rate per model; engaged/locked verdict
- `analysis/pooled.py` -- pooled binary+ternary conditional logit (parametric
  and non-parametric); LR test on pooling validity
- `analysis/ternary.py` -- ternary-only conditional logit for the
  binary-vs-ternary forest plot
- `analysis/personality.py` -- quality coefficients and capability-scatter inputs
- `analysis/brand.py` -- chain-family coefficients with full controls;
  permutation test on within-provider correlation
- `analysis/figures.py` -- renders all figures from the CSV outputs above

Outputs land in `web/data/` and `web/figures/`.

### Public writeup
- `web/index.html` -- standalone landing page with full results, methodology,
  references, and replication appendix
- `web/style.css` -- styles
- `web/data/`, `web/figures/` -- data and figures rendered by the pipeline

## How to add a new LLM

1. Score the model twice (original + swap) using whichever runner fits its API.
2. Add one entry to the `MODELS` dict in `config.py` mapping the display name
   to the (original, swap) CSV pair.
3. Re-run the analysis pipeline:
   `triage.py` -> `pooled.py` -> `ternary.py` -> `personality.py` -> `brand.py` -> `figures.py`.
   All scripts iterate over `MODELS` automatically.

## How to vary prompts

The prompt template lives in `run_conjoint_llm.py` as `BINARY_PROMPT_TEMPLATE`
and `TERNARY_PROMPT_TEMPLATE`. To test a variant:

1. Edit the template, or copy the script with the alternate prompt.
2. Use `--tag` to distinguish output files:
   ```
   python run_conjoint_llm.py --model gemma3:latest --tag prompt_v2
   python run_conjoint_llm.py --model gemma3:latest --tag prompt_v2 --swap
   ```
3. Register the variant as a separate entry in `config.py`:
   ```python
   'Gemma3 4B (v2 prompt)': ('conjoint_results_gemma3_latest_prompt_v2.csv',
                              'conjoint_results_gemma3_latest_prompt_v2_swap.csv'),
   ```

## Running scripts

All scripts run from the project root:
```
/c/Users/pak_o/anaconda3/envs/datasci/python.exe <script.py>
```

## Conventions

- Text output uses UTF-8 encoding (pass `encoding='utf-8'` when writing)
- ASCII-only characters in formatted output (no Unicode box-drawing --
  breaks cp1252 console)
- Plots save at 200 DPI as PNG
- File naming: `conjoint_results_{model_tag}[_{extra_tag}][_swap].csv`
- Swap task IDs are offset by +10000
