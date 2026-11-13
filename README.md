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

GPU runs on Google Colab (free T4): `python scripts/make_colab_bundle.py` writes
`dist/codeguard_colab_bundle.zip` (code + both data tiers, no raw data, weights or runs). Upload it to
`MyDrive/CodeGuard/`, open `notebooks/colab_runs.ipynb` in Colab with a T4 runtime and use Run all: tests,
preflight with timing, the `SMOKE=1` server rehearsal, the two small-tier insurance runs and, if time
remains, `zs_6.7b_small_4bit`. Outputs are copied to `MyDrive/CodeGuard/out/` after every stage and finished
stages are skipped on a re-run.

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

**Repair, small tier (Colab T4, 2026-11-13).** Source: `results/results.csv`, `results/headline.json`.
Selected on val by exact match before test was read: `lora_1.3b_r16_small` (`results/selection.json`).
95% bootstrap intervals over 500 pairs in brackets. No 3-shot run on the small tier.

Test, headline runs (DeepSeek-Coder-1.3B-instruct):

| Run | Exact match | Localized exact | Localized | Fix direction | Parse ok | Changed | BLEU (copy 93.2) |
|---|---|---|---|---|---|---|---|
| `zs_1.3b_small` (zero-shot) | 0.034 [0.018, 0.050] | 0.062 [0.042, 0.084] | 0.580 [0.538, 0.624] | 0.564 [0.520, 0.608] | 0.554 | 0.888 | 78.6 |
| `lora_1.3b_r16_small` (LoRA r16) | 0.132 [0.104, 0.164] | 0.436 [0.392, 0.480] | 0.456 [0.412, 0.500] | 0.558 [0.514, 0.600] | 0.990 | 0.966 | 89.8 |
| LoRA minus zero-shot (paired) | +0.098 [0.072, 0.126] | +0.374 [0.326, 0.418] | −0.124 [−0.184, −0.064] | −0.006 [−0.034, 0.020] | | | |

Val, all repair runs:

| Run | Exact match | Localized exact | Localized | Fix direction | Parse ok |
|---|---|---|---|---|---|
| `zs_1.3b_small` | 0.018 [0.008, 0.030] | 0.042 [0.026, 0.060] | 0.598 [0.556, 0.640] | 0.500 [0.456, 0.546] | 0.468 |
| `zs_6.7b_small_4bit` (6.7B zero-shot, 4-bit) | 0.016 [0.006, 0.028] | 0.056 [0.038, 0.078] | 0.750 [0.712, 0.786] | 0.530 [0.488, 0.574] | 0.444 |
| `lora_1.3b_r16_small` | 0.118 [0.090, 0.148] | 0.450 [0.406, 0.496] | 0.478 [0.434, 0.522] | 0.492 [0.446, 0.534] | 0.990 |

Test numbers of `zs_6.7b_small_4bit` (not used for selection): exact match 0.032 [0.018, 0.048],
localized exact 0.068 [0.048, 0.090], fix direction 0.580 [0.538, 0.622].

What this shows, and what it does not:

- LoRA on 4000 pairs raises exact match from 3.4% to 13.2% on test and makes the model edit exactly the
  lines the developer edited 44% of the time instead of 6%. Most bugs (87%) are still not fixed exactly.
- Fix direction does not change with LoRA and stays close to chance (0.5): val 0.49–0.53, test 0.56–0.58.
- "Localized" (any overlap) drops with LoRA because the zero-shot model rewrites many lines, which overlaps
  the developer's line by accident; "localized exact" is the informative one.
- The zero-shot model writes an explanation before the code, and the generation budget (1.5 × input
  tokens + 32) often ends inside the code block: only 47–55% of zero-shot outputs have a closed code block,
  and an unclosed one never matches. A post-hoc check on the pairs where the zero-shot output is well-formed
  (234 val / 277 test pairs) still gives LoRA +0.073 [0.030, 0.111] (val) and +0.079 [0.043, 0.112] (test)
  exact match over zero-shot, so the gain is not only output format.
- The 6.7B model zero-shot in 4-bit is no better than 1.3B zero-shot on exact match.
- LoRA BLEU (89.8) is below the copy baseline (93.2): BLEU does not separate good from bad repairs here.
- One seed, small tier, 1.3B model only.

Cost (`results/costs.csv`): 14,991,360 trainable parameters (1.1% of 1.36B), adapter 59 MB, 2 epochs over
3991 pairs in 35 min at 3.79 examples/s on a T4 (fp16). Figures: `results/figures/headline_test.png`,
`by_pattern_test.png`, `loss_curve.png`, `detection_val.png`; error-analysis sample:
`results/error_analysis.csv` (notes to be written by the students).

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
