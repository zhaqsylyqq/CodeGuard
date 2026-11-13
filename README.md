# CodeGuard

LoRA / QLoRA fine-tuning of a code LLM (DeepSeek-Coder-instruct) to **repair single-statement Python bugs**:
given a buggy code fragment, generate the fixed fragment. Detection baselines ("is this snippet buggy?") are
kept as a secondary track. Data: CTSSB-1M, about 1M single-statement bug fixes mined from GitHub
([TSSB-3M](https://github.com/cedricrupb/TSSB3M), cleaned version).

Course project: Introduction to NLP (CSCI1407), KBTU, Fall 2026.
Authors: Zhaxylyk Abdimutalipov, Alrami Dandibay.

Design, experiment matrix and acceptance criteria: [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md).
Team guide in Russian: [docs/PROJECT_GUIDE_RU.md](docs/PROJECT_GUIDE_RU.md).
Server window: [docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md).

## Why repair and not detection

Snippets are diff hunks of about 7 lines and the buggy and fixed versions differ by one statement. Shown one
snippet without the rest of the file, "buggy or fixed?" is mostly not answerable: both trained detection
baselines sit at chance (table below). Repair is well-posed on the same data and is checked against the real
developer fix.

## Layout

```
data/raw/                 CTSSB-1M shards (not in git)
data/processed/small/     train/val/test.parquet, 4000/500/500 pairs (not in git)
data/processed/full/      train/val/test.parquet, 50000/2000/5000 pairs; small is a subset (not in git)
notebooks/
  01_data.ipynb           data preparation, both tiers, EDA plots
  02_baseline.ipynb       TF-IDF + logistic regression (detection)
  03-05                   BiLSTM, LLM zero-shot and LoRA detection (legacy track)
  06_results.ipynb        figures, error analysis, cost table from runs/ and results/
src/
  data_utils.py           loading, diff -> code, dedup, project split, nested sampling, length filter
  experiment_utils.py     load_pairs, bootstrap intervals, run metadata
  llm_utils.py            model loading (bf16 / 4-bit / multi-GPU), detection prompt and scoring
  repair_utils.py         repair prompt, label masking, generation, parsing, repair metrics
  train_repair.py         LoRA / QLoRA training for repair, then evaluation on val and test
  eval_repair.py          zero-shot / 3-shot / adapter evaluation
  make_report.py          runs/ -> results/*.csv, selection.json (chosen on val only)
  bilstm.py, llm_baseline.py, train_lora.py   detection track
scripts/
  download_models.py      pre-fetch weights; print pinned revisions
  preflight.py            environment / GPU / data / model checks and a timing run
  run_server.sh           the whole server window in one command (one queue per GPU)
app/                      FastAPI demo (server.py + index.html)
tests/                    unit and smoke tests on tiny models and synthetic fixtures
results/                  summary.json, baseline metrics, figures; report tables after real runs
runs/                     one directory per run: metrics.json, predictions, adapter (not in git)
```

## How to run (small tier, Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # CUDA build
python -m pip install -r requirements-neural.txt

# raw data: data/raw/ctssb/ctssb_data_1M/file-0.jsonl.gz ... file-9.jsonl.gz
python -m pytest tests -q
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data.ipynb   # builds both tiers
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline.ipynb
python src/bilstm.py --output runs/bilstm_small
```

Repair runs on the small tier with the 1.3B model (one GPU; set `CUDA_VISIBLE_DEVICES` if you have several):

```powershell
$M = "deepseek-ai/deepseek-coder-1.3b-instruct"; $REV = "e063262dac8366fc1f28a4da0ff3c50ea66259ca"
python src/eval_repair.py  --model $M --revision $REV --output runs/zs_1.3b_small --shots 0
python src/train_repair.py --model $M --revision $REV --output runs/lora_1.3b_r16_small --rank 16 `
    --batch-size 2 --accumulation-steps 16
python src/make_report.py --runs runs --results results
```

Add `--limit-pairs 16` for a quick smoke run (write it to a separate output directory). Every script writes
`metrics.json` last and skips the run if it already exists, so re-running a command after a crash continues
where it stopped. Data defaults to `data/processed/small`; pass `--data-dir data/processed/full` for the full
tier. The server plan (6.7B, rank ablation, QLoRA, 33B) is run by `scripts/run_server.sh`; see the runbook.

Demo: `python app/server.py`, then open http://127.0.0.1:8000 (uses the small-tier adapter if it exists).

## Data

| Step | Pairs |
|---|---|
| Raw records | 966,985 |
| `in_function` | 714,241 |
| Single hunk | 714,241 |
| Real change (buggy != fixed) | 713,790 |
| After deduplication | 647,850 |
| After removing cross-split duplicates | 647,683 |
| After length filter (<= 2000 chars and <= 60 lines) | 647,571 |
| Small tier (train / val / test) | 4000 / 500 / 500 |
| Full tier (train / val / test) | 50,000 / 2000 / 5000 |

Splits are by project (80/10/10); no project or snippet is shared between splits in the pool or in either
tier, and small is a strict subset of full within every split. Python 2 share: 4.9% of the pool.
Source: `results/summary.json`.

## Results

Only numbers produced by a run with a `metrics.json` are listed. Validation split, small tier.

**Detection (isolated snippet: buggy or fixed?)**

| Model | Accuracy | F1 | ROC-AUC | Source |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.523 | 0.436 | – | `results/baseline_metrics.json` |
| BiLSTM | 0.499 | 0.209 | 0.4998 | `results/bilstm_metrics.json` |

Both are at chance level (classes are 50/50; with 1000 examples chance spans roughly 0.47–0.53).

**Repair**: no completed runs yet. The table is filled from `results/results.csv` after real runs
(`python src/make_report.py`). Test numbers are reported only for the configuration selected on validation
(see `results/selection.json`).

## Evaluation protocol in one paragraph

Every run writes predictions for val and test. All comparisons (rank, quantization, model size) are made on
val; `make_report.py` picks the final configuration by val exact match and writes that choice to
`results/selection.json` before it reads any test number. The headline table shows test numbers for zero-shot,
3-shot and the selected LoRA model, with 95% bootstrap intervals and paired bootstrap differences.
Metrics: exact match after whitespace normalization (headline), localization (did the model edit the line the
developer edited), copy rate, BLEU next to the copy baseline, and fix-direction accuracy
(`log P(fixed | buggy) > log P(buggy | fixed)`, chance 0.5).

## Limitations

Single-statement bugs only; fragments without file context; exact match undercounts alternative correct fixes;
4.9% Python 2 code; deduplication is exact-match only; one seed unless the optional seed runs are done;
the 33B reference model is evaluated in 4-bit.
