# CodeGuard

Detecting bugs in Python code: given a code snippet, decide whether it is **buggy** or **fixed**.
Data: CTSSB-1M, about 1M single-statement bug fixes mined from GitHub
([TSSB-3M](https://github.com/cedricrupb/TSSB3M), cleaned version).

Course project: Introduction to NLP (CSCI1407), KBTU, Fall 2026.
Authors: Zhaxylyk Abdimutalipov, Alrami Dandibay.

## Project layout

```
data/raw/            original CTSSB-1M shards (not in git)
data/processed/      pool.parquet, train/val/test.parquet (not in git)
notebooks/
  01_data.ipynb      data preparation + EDA plots
  02_baseline.ipynb  TF-IDF + logistic regression baseline
src/data_utils.py    reusable data functions
tests/               unit tests for data_utils
results/             summary.json and figures
```

## How to run

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

# put ctssb_data_1M.zip content into data/raw/ctssb/ctssb_data_1M/ (file-0.jsonl.gz ... file-9.jsonl.gz)
python -m pytest tests          # unit tests
jupyter lab                     # then run notebooks/01_data.ipynb top to bottom
```

The first run reads all shards (about 2 minutes) and caches the result in `data/processed/pool.parquet`.

## What the data pipeline does

1. Load all shards, keep rows with `in_function == True`.
2. Rebuild `buggy` and `fixed` code from the diff; keep diffs with a single hunk.
3. Drop pairs where buggy equals fixed after normalization (indentation is preserved), then exact duplicates.
4. Flag Python 2 code (not removed).
5. Split 80/10/10 by `project`, so no project appears in two splits.
6. Remove val/test pairs whose code also appears in an earlier split.
7. Sample 4000 / 500 / 500 pairs (seed 42) and save `train/val/test.parquet`.

## Key numbers

| Step | Pairs |
|---|---|
| Raw records | 966,985 |
| `in_function` | 714,241 |
| Single hunk | 714,241 |
| Real change (buggy != fixed) | 713,790 |
| After deduplication | 647,850 |
| After removing cross-split duplicates | 647,683 |
| Sample (train / val / test) | 4000 / 500 / 500 |

Python 2 share: 4.9% of the pool, 5.1% of the sample. No project or snippet is shared between splits.
Full numbers: `results/summary.json`.

## Baseline (TF-IDF + Logistic Regression, validation set)

| Metric | Value |
|---|---|
| Accuracy | 0.503 |
| Precision | 0.506 |
| Recall | 0.274 |
| F1 | 0.355 |

The baseline is at chance level (classes are 50/50): buggy and fixed snippets share almost all tokens,
so a bag-of-tokens model cannot tell them apart. Details: `notebooks/02_baseline.ipynb`, `results/baseline_metrics.json`.
