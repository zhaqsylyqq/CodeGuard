# CodeGuard roadmap

Status table aligned with [AGENT_GUIDE.md](AGENT_GUIDE.md) (which wins on any conflict). "Run" means executed
on real data with a `metrics.json` (or, for data, the executed notebook); "code-checked" means tests and smoke
runs only.

| Work package | Implementation | Status (2026-11-14) |
|---|---|---|
| WP1 Data tiers | `data_utils.sample_nested`, length filter, `01_data.ipynb` | Run: small 4000/500/500, full 50000/2000/5000, zero leakage |
| WP2 Repair core | `repair_utils.py`, `experiment_utils.py` | Code-checked (unit tests; real DeepSeek tokenizer for masking) |
| WP3 Train / eval scripts | `train_repair.py`, `eval_repair.py` | Run: both small-tier insurance runs on a Colab T4 (metrics.json) |
| WP4 Detection track | `bilstm.py`, `llm_baseline.py`, `train_lora.py`, notebooks 02–05 | TF-IDF and BiLSTM run on small tier (chance level); LLM detection not run |
| WP5 Server tooling | `scripts/`, `docs/SERVER_RUNBOOK.md` | Preflight with model checks and timing passed on a Colab T4 (Linux); `SMOKE=1` rehearsal still not done (the Colab output has no trace of that stage) |
| WP6 Reporting | `make_report.py`, `06_results.ipynb` | Run on the three small-tier repair runs; figures 2-3 wait for the server runs |
| WP7 Demo | `app/` | Code-checked with a tiny model; not yet run with real weights |
| WP8 Docs | README, this file, `PROJECT_GUIDE_RU.md` | Small-tier results filled from `results/`; server results pending |

## Experiment matrix (section 8 of the guide)

| Run | Status |
|---|---|
| `zs_1.3b_small`, `lora_1.3b_r16_small` (insurance, small tier) | Done on a Colab T4 (2026-11-13); test exact match 0.034 -> 0.132 (README) |
| `zs_6.7b_small_4bit` (6.7B zero-shot, 4-bit, small tier) | Done on a Colab T4; val exact match 0.016 |
| R1–R8 (must) | Waiting for the server window |
| R9–R10 (should) | Waiting for the server window |
| R11 `zs_33b`, R12 seeds 43/44 (optional) | Only if time remains |

## Before the server window

1. Ask the instructor the questions at the top of `docs/SERVER_RUNBOOK.md`.
2. ~~Finish both insurance runs and run `make_report.py` on them.~~ Done.
3. Download the 6.7B weights (`scripts/download_models.py`) and copy the HF cache if the server has no internet.
4. Rehearse `SMOKE=1 GPUS="0" bash scripts/run_server.sh` on a Linux machine with a GPU (stage 6 of `notebooks/colab_runs.ipynb`).
5. Run `scripts/preflight.py` with the timing run to choose `BATCH` / `ACCUM`.
6. Decide the training length and evaluation size for the window (next section).

## Measured speed and what it means for the server plan

Measured on a Colab Tesla T4 (fp16; `logs/preflight_colab.log`, `runs/*/metrics.json`):

| What | Speed |
|---|---|
| LoRA r16 training, 1.3B, max length 768, batch 16 (preflight, 30 steps) | 4.4 examples/s; preflight projects 382 min for 50k pairs x 2 epochs on a T4 |
| Same, real run, batch 8 x 4, with 8 in-training evaluations | 3.79 examples/s (2 epochs x 3991 pairs in 35 min) |
| Generation + direction scoring, 1.3B, batch 16 | 1.3 pairs/s (LoRA), 1.9 pairs/s (zero-shot) |
| Same, 6.7B in 4-bit, batch 8 | 0.33 pairs/s |
| Small-tier validation loss | lowest after epoch 1 (0.1054), flat in epoch 2 |

Rough projection for one RTX 4090. **These are estimates, not measurements.** They assume the 6.7B model
costs about 5x the 1.3B model per example (parameter ratio) and that a 4090 is 3-5x faster than a T4. The
server preflight times the 6.7B model at the start of the window and replaces these numbers.

- 6.7B LoRA training: about 2.5-4.5 examples/s, so 50k pairs x 2 epochs (100k examples) needs about 6-11 h.
  `BUDGET_MIN=150` stops a run after about 22k-40k examples, i.e. 0.45-0.8 of one epoch. With
  `--epochs 2` the cosine schedule is planned for 2 epochs, so a run cut at half an epoch stops with the
  learning rate still near its peak. Option: set `--epochs` (or the number of training pairs) to what
  fits the budget, so the schedule finishes.
- 1.3B LoRA training (R7): about 13-22 examples/s, so 50k x 2 epochs takes about 1.3-2.1 h and probably fits.
- Evaluation is the larger risk. Every run predicts val (2000) + test (5000) = 7000 pairs. At a 3-5x
  speed-up over the T4 4-bit number, a 6.7B run needs about 1.2-2 h of evaluation, far more than
  `EVAL_MIN=45`. GPU 5 runs three evaluation-only 6.7B/1.3B runs in a row. Option: a fixed, seeded
  evaluation subset (for example 1000 val + 2000 test pairs) for the server runs. This needs a separate
  evaluation limit in the scripts, because `--limit-pairs` also limits the training pairs.
- The zero-shot generation budget (1.5 x input tokens + 32) cuts off about half of the zero-shot answers
  inside the code block (small tier). For R1-R3 consider a larger budget for untrained runs, or report the
  well-formed subset next to the headline.

## After the window

Run `make_report.py`, execute `06_results.ipynb`, fill README and the Russian guide from `results/` only, and
write the error-analysis notes.

## Later extensions (out of scope now)

- Near-duplicate detection beyond exact hashes.
- Other model families via `--model` (e.g. a Qwen coder model) after a compatibility check.
- File-level context in the prompt.
