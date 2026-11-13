# CodeGuard roadmap

Status table aligned with [AGENT_GUIDE.md](AGENT_GUIDE.md) (which wins on any conflict). "Run" means executed
on real data with a `metrics.json` (or, for data, the executed notebook); "code-checked" means tests and smoke
runs only.

| Work package | Implementation | Status (2026-11-13) |
|---|---|---|
| WP1 Data tiers | `data_utils.sample_nested`, length filter, `01_data.ipynb` | Run: small 4000/500/500, full 50000/2000/5000, zero leakage |
| WP2 Repair core | `repair_utils.py`, `experiment_utils.py` | Code-checked (unit tests; real DeepSeek tokenizer for masking) |
| WP3 Train / eval scripts | `train_repair.py`, `eval_repair.py` | Code-checked; smoke on real 1.3B on a 4 GB GPU; insurance runs in progress |
| WP4 Detection track | `bilstm.py`, `llm_baseline.py`, `train_lora.py`, notebooks 02–05 | TF-IDF and BiLSTM run on small tier (chance level); LLM detection not run |
| WP5 Server tooling | `scripts/`, `docs/SERVER_RUNBOOK.md` | Code-checked (plans for 6/2/1 GPUs, preflight without model checks); `SMOKE=1` rehearsal on a Linux GPU machine not done |
| WP6 Reporting | `make_report.py`, `06_results.ipynb` | Code-checked on fake fixtures only |
| WP7 Demo | `app/` | Code-checked with a tiny model; not yet run with real weights |
| WP8 Docs | README, this file, `PROJECT_GUIDE_RU.md` | Done; results tables to be filled from real runs |

## Experiment matrix (section 8 of the guide)

| Run | Status |
|---|---|
| `zs_1.3b_small`, `lora_1.3b_r16_small` (insurance, small tier, local GPU) | In progress |
| R1–R8 (must) | Waiting for the server window |
| R9–R10 (should) | Waiting for the server window |
| R11 `zs_33b`, R12 seeds 43/44 (optional) | Only if time remains |

## Before the server window

1. Ask the instructor the questions at the top of `docs/SERVER_RUNBOOK.md`.
2. Finish both insurance runs and run `make_report.py` on them.
3. Download the 6.7B weights (`scripts/download_models.py`) and copy the HF cache if the server has no internet.
4. Rehearse `SMOKE=1 GPUS="0" bash scripts/run_server.sh` on a Linux machine with a GPU.
5. Run `scripts/preflight.py` with the timing run to choose `BATCH` / `ACCUM`.

## After the window

Run `make_report.py`, execute `06_results.ipynb`, fill README and the Russian guide from `results/` only, and
write the error-analysis notes.

## Later extensions (out of scope now)

- Near-duplicate detection beyond exact hashes.
- Other model families via `--model` (e.g. a Qwen coder model) after a compatibility check.
- File-level context in the prompt.
