# CodeGuard: architecture and build guide for coding agents

Audience: Claude Code agents that will implement the rest of this project, and the two students who
own it (Zhaxylyk Abdimutalipov, Alrami Dandibay). Written 2026-11-12 against commit `8526c5a`.

Read this whole file before changing anything. It is the single source of truth for scope, design
and acceptance criteria. Where it conflicts with `docs/roadmap.md`, this file wins; update the
roadmap to match (work package WP8).

---

## 1. Context

**Course project.** Introduction to NLP (CSCI1407), KBTU, Fall 2026. Topic: parameter-efficient
fine-tuning (LoRA / QLoRA) of a code LLM for Python bugs. Endterm demo and report are roughly three
to four weeks away. The instructor knows AI tools write the implementation; the requirement on us is
that the students understand and can defend every design decision. So: small readable code, plain
explanations, no architecture for its own sake.

**What exists and works (do not rewrite).**

- `src/data_utils.py` + `notebooks/01_data.ipynb`: CTSSB-1M → (buggy, fixed) snippet pairs, split
  80/10/10 **by project**, cross-split duplicates removed, zero leakage verified. Pool after
  cleaning: 647,683 pairs (train 517,301 / val 63,501 / test 66,881). Current sample: 4000/500/500.
- `notebooks/02_baseline.ipynb`: TF-IDF + logistic regression, val accuracy 0.503.
- `src/bilstm.py`: BiLSTM, val accuracy 0.500, ROC-AUC 0.4998.
- `src/llm_utils.py`, `src/llm_baseline.py`, `src/train_lora.py`: answer-likelihood classification
  ("buggy" vs "fixed") for a causal LM, zero-shot and with LoRA. Implemented, never run on real data.
- Tests in `tests/` (synthetic fixtures; no real weights needed).

**Why the scope changes.** The current task is "given ONE snippet, say buggy or fixed". Snippets are
diff hunks of about 7 lines (median 21 whitespace tokens) and the two versions differ by a single
statement; 15% of pairs are a changed string literal. Without the rest of the file the label is
mostly not recoverable, and both trained baselines sit exactly at chance. A LoRA model on this
framing would most likely land a few points above 50%, which is a weak result for a one-shot window
on a 6-GPU server. This is a forecast, not a measurement; one cheap run (R10 below) will measure it.

**New primary task: single-statement bug repair.** Input: the buggy snippet. Output: the fixed
snippet. Same data, well-posed, verifiable against the real developer fix, and it is the
"generate a fix recommendation" half of the original topic. Detection stays in the report as (a) the
honest chance-level baselines and (b) a pair-level "fix direction" metric computed from the same
repair model.

**Research questions the final report must answer.**

1. How much does LoRA improve repair over the same model zero-shot and 3-shot?
2. How does LoRA rank (8/16/32) affect it?
3. How does model size (1.3B vs 6.7B) affect it?
4. What does 4-bit quantization (QLoRA) cost against full-precision LoRA?
5. Does a small LoRA-tuned model beat a much larger untuned model (33B zero-shot)? (optional)
6. Which bug types (`sstub_pattern`) are fixable and which are not, and why?
7. Is isolated single-snippet detection learnable at all? (baselines + one LoRA run)

---

## 2. Rules for agents

These are hard constraints.

1. **Never fabricate results.** No invented metrics in README, docs, notebooks or commit messages.
   A number may be written down only if it was produced by a run whose `metrics.json` exists.
   Synthetic-fixture checks are code checks, not results; never present them as results.
2. **Test split discipline.** Nothing is selected, tuned or early-stopped on test. All choices use
   val. See section 6.5 for the exact protocol.
3. **Keep it small.** Flat `src/` with plain modules and `argparse`. No package/pyproject, no
   config framework, no registries, no experiment-tracking service, no Docker, no distributed
   training. If a new abstraction is not needed by two call sites, do not add it.
4. **Do not break the data pipeline.** `rebuild_snippets`, `normalize_code` (indentation is
   preserved on purpose), the project-grouped split and the leakage checks stay as they are.
   Extending is fine; changing behaviour of existing functions needs a test that shows why.
5. **Verify library APIs instead of assuming.** Versions of `transformers`, `peft`, `bitsandbytes`
   move fast. Before using an argument or behaviour you are not sure of (chat templates, EOS token of
   a given tokenizer, `Trainer` arguments, multi-GPU behaviour), check the installed version's docs
   or source, or write a tiny test. Pin what you verified in `requirements-neural.txt`.
6. **Cross-platform.** Students develop on Windows (PowerShell, `.venv`); the server is assumed to
   be Linux. All Python must run on both. Only `scripts/run_server.sh` may be bash-only.
7. **Every stage has a smoke path** that runs in minutes on CPU or one small GPU
   (`--limit-pairs`, tiny model). A stage is not done until its smoke path and its tests pass.
8. **Heavy runs are not yours to fake or to launch blindly.** If you have no GPU or no real data,
   implement, test on fixtures, and state plainly in your final message what was not run.
   `data/raw` and `data/processed` are not in git; they live on the students' machines.
9. **Docs are part of the deliverable.** Code/README in English. `docs/PROJECT_GUIDE_RU.md` in plain
   Russian for the team, updated for everything you add (WP8).
10. **Git.** Work on a feature branch, small commits, clear messages, never commit data, weights or
    `runs/`. Do not rewrite history on `main`.

---

## 3. Target layout

New files are marked NEW, changed ones CHG. Nothing else should appear in `src/`.

```
CLAUDE.md                      short pointer to this guide
docs/AGENT_GUIDE.md            this file
docs/PROJECT_GUIDE_RU.md  CHG  team guide, plain Russian
docs/roadmap.md           CHG  status table aligned with this guide
docs/SERVER_RUNBOOK.md    NEW  one page: how to run the server window, for whoever runs it
notebooks/
  01_data.ipynb           CHG  nested samples: small + full tiers, length filter
  02_baseline.ipynb       CHG  reads the small tier path
  03..05                       unchanged apart from data path
  06_results.ipynb        NEW  tables, figures, error analysis from runs/
src/
  data_utils.py           CHG  + sample_nested, + length filter
  experiment_utils.py     CHG  + load_pairs, + bootstrap_ci, + run metadata
  llm_utils.py            CHG  + dtype/device options; classification code untouched in behaviour
  repair_utils.py         NEW  prompt, encoding, generation, parsing, repair metrics
  train_repair.py         NEW  LoRA / QLoRA training for repair
  eval_repair.py          NEW  zero-shot / few-shot / adapter evaluation
  make_report.py          NEW  runs/ -> results/results.csv + selection of the final model
  bilstm.py, llm_baseline.py, train_lora.py   (legacy detection track; minimal changes)
scripts/
  download_models.py      NEW  pre-fetch weights into the HF cache
  preflight.py            NEW  environment/GPU/data/model checks before the window
  run_server.sh           NEW  the whole server window, one command
app/
  server.py               NEW  FastAPI demo
  index.html              NEW  single static page
tests/
  test_repair_utils.py    NEW
  test_report.py          NEW
```

---

## 4. Data (WP1)

### 4.1 Two nested tiers

| Tier | Path | train / val / test pairs | Used for |
|---|---|---|---|
| small | `data/processed/small/` | 4000 / 500 / 500 | CPU baselines, debugging, Colab |
| full | `data/processed/full/` | 50,000 / 2000 / 5000 | server runs, final numbers |

Requirements:

- Add `sample_nested(df, sizes_by_tier, seed)` to `data_utils.py`: shuffle each split once with the
  seed, then take prefixes, so that **small ⊂ full** within every split. Test this property.
- Apply one extra filter before sampling, counted in the funnel: drop pairs where either snippet is
  longer than 2000 characters or 60 lines (they are rare outliers; max seen is 346 tokens). Report
  how many were dropped. The filter is tokenizer-independent on purpose.
- Keep saving `project, file_path, sstub_pattern, buggy, fixed, is_py2, buggy_hash, fixed_hash,
  split`. Add a stable `pair_id` column (the row's index in the cleaned pool) so predictions from
  different runs can be joined.
- Re-run `check_leakage` on both tiers and store everything in `results/summary.json`.
- The old flat `data/processed/{train,val,test}.parquet` paths go away; every script takes
  `--data-dir` with default `data/processed/small`.
- TF-IDF and BiLSTM are re-run on the new small tier (CPU, minutes) so all reported numbers come
  from one consistent sample. Their old numbers (0.503 / 0.500) are expected to stay near chance; if
  they do not, stop and investigate before continuing.

The 5000-pair test set gives roughly ±1.4 percentage points at 95% for a proportion near 50% and
tighter near 10–20%, which is enough to separate the configurations we compare.

### 4.2 Where the raw data is

CTSSB-1M shards (`file-0.jsonl.gz` … `file-9.jsonl.gz`) are on the students' machines under
`data/raw/ctssb/ctssb_data_1M/`. If you run somewhere without them: implement and test on synthetic
fixtures, and say so. Do not substitute another dataset.

---

## 5. Repair task design (WP2, WP3)

### 5.1 Prompt

One prompt for zero-shot, few-shot, training and the demo. Built with the tokenizer's own chat
template (`tokenizer.apply_chat_template`) so each model gets its native format.

User message:

````
Fix the bug in the following Python code. It is a fragment of a larger file and contains exactly one buggy statement. Return the complete corrected fragment in a single Python code block and nothing else.

```python
{buggy}
```
````

Assistant message (training target):

````
```python
{fixed}
```
````

- Training labels cover only the assistant message plus the end-of-turn token; prompt tokens are
  masked with -100. Write a test that decodes the unmasked label span of one example and checks it
  equals the target text and ends with the token generation actually stops on. Find that token from
  the tokenizer/chat template of the real model; do not hard-code it from memory.
- Few-shot (k=3): three fixed train pairs (chosen once with the seed, different `sstub_pattern`s,
  short) are prepended as earlier user/assistant turns. Same three for every test item; store their
  `pair_id`s in `metrics.json`.
- Parsing the output: take the content of the first fenced code block; if there is no fence, take
  the raw text. Strip exactly one leading and trailing newline, never leading spaces of the first
  line (indentation is meaningful). Record `parse_ok`.
- No truncation of the code. Examples whose prompt + target exceed `--max-length` (default 768
  tokens) are dropped from training and counted. For evaluation nothing is dropped: the prompt is
  never cut (the length filter in 4.1 guarantees it fits), and `max_new_tokens` =
  `min(512, 1.5 × input snippet tokens + 32)`.

### 5.2 Models

| Role | Hugging Face id |
|---|---|
| debug / small | `deepseek-ai/deepseek-coder-1.3b-instruct` |
| main | `deepseek-ai/deepseek-coder-6.7b-instruct` |
| large reference, zero-shot only, optional | `deepseek-ai/deepseek-coder-33b-instruct` |

Use the instruct variants (the existing default is `-base`, which does not follow instructions and
would make the zero-shot baseline unfairly weak). Model id is always a `--model` flag, so another
family (for example a Qwen coder model) can be tried without code changes; it is not in scope.
Pin `--revision` for every model in `scripts/run_server.sh` after the first download and record it
in `metrics.json`.

### 5.3 LoRA configuration

| Setting | Value | Note |
|---|---|---|
| target modules | all linear projections of the decoder blocks (attention q/k/v/o and MLP) | resolve names from the model, do not hard-code blindly |
| rank r | 16 main; 8 and 32 ablation | |
| alpha | 2 × r | |
| dropout | 0.05 | |
| precision | bf16, no quantization (main); 4-bit NF4 + bf16 compute (QLoRA run) | fall back to fp16 only if bf16 is unsupported |
| optimizer | AdamW, lr 2e-4, cosine schedule, 3% warmup, weight decay 0 | |
| effective batch | 32 (per-device batch × accumulation) | per-device batch chosen by the timing run |
| epochs | 2 | capped by the time budget, see below |
| max length | 768 tokens | |
| length grouping | on | less padding |
| seed | 42 | |

### 5.4 `train_repair.py`

- Args: `--model --revision --data-dir --output --rank --quantized --epochs --batch-size
  --accumulation-steps --lr --max-length --limit-pairs --seed --time-budget-min --eval-pairs`.
- **Time budget.** `--time-budget-min N` installs a small `TrainerCallback` that stops training
  cleanly once N minutes have elapsed, after which the script still saves the adapter and runs the
  evaluation. The server window is fixed; a run must never be killed with nothing saved.
- During training: `eval_loss` on a fixed 500-pair subset of val, four times per epoch; keep the
  best checkpoint by `eval_loss`; `save_total_limit=2`.
- **Resumable and idempotent.** If `output/metrics.json` exists, exit 0 immediately. If a checkpoint
  exists, resume from it.
- After training: save adapter + tokenizer to `output/adapter`, then call the same evaluation
  function as `eval_repair.py` on val and on test (section 6.5 explains why test is computed here).
- Must run on exactly one GPU. If more than one GPU is visible, exit with a clear message telling
  the user to set `CUDA_VISIBLE_DEVICES`. (`Trainer` would otherwise try to use all of them.)

### 5.5 `eval_repair.py`

- Args: `--model --revision --adapter --shots {0,3} --quantized --data-dir --splits val test
  --output --batch-size --limit-pairs --seed --multi-gpu`.
- Greedy decoding (no sampling), batched, left padding for generation, inputs sorted by length.
- `--multi-gpu` loads with `device_map="auto"` across the visible GPUs. Only the 33B zero-shot run
  uses it.
- Also computes the fix-direction metric (6.2) unless `--no-direction`.
- Writes, per split: `output/{split}_predictions.parquet` with `pair_id, sstub_pattern, is_py2,
  prediction, parse_ok, exact_match, localized, localized_exact, changed, direction_correct,
  n_new_tokens`; and one `output/metrics.json`.

---

## 6. Evaluation (WP2, WP6)

### 6.1 Repair metrics (all in `repair_utils.py`, pure functions, unit-tested)

Let `norm = data_utils.normalize_code`.

| Metric | Definition |
|---|---|
| `exact_match` (headline) | `norm(prediction) == norm(fixed)` |
| `changed` | `norm(prediction) != norm(buggy)`; `1 - changed` is the "just copied the input" rate |
| `localized` | the buggy lines the model edited overlap the buggy lines the developer edited |
| `localized_exact` | the two edited-line sets are equal |
| `bleu` | corpus BLEU (sacrebleu) of prediction vs fixed; **always reported next to the copy baseline** (BLEU of buggy vs fixed), because the two versions share almost all tokens and BLEU alone is misleading |
| `parse_ok` rate | share of outputs with a well-formed code block |

Edited-line sets: `difflib.SequenceMatcher(None, buggy_lines, other_lines, autojunk=False)`
on normalized lines; for every non-`equal` opcode add `range(i1, max(i2, i1 + 1))` (so a pure
insertion counts as touching the line it is inserted before). `localized` is false when `changed`
is false.

CodeBLEU is deliberately not used: it needs parseable code and data-flow graphs, and our snippets
are fragments that mostly do not parse (see limitation 5 in the Russian guide). State this in the
report.

Breakdowns stored in `metrics.json`: every metric overall, per `sstub_pattern` (patterns with fewer
than 30 test pairs grouped as `OTHER`), and for `is_py2` true/false.

### 6.2 Fix-direction accuracy (pair-level detection from the repair model)

For each pair compute, with the repair prompt:

`s_fwd = log P(fixed | prompt(buggy))`, `s_bwd = log P(buggy | prompt(fixed))`

(sum of target-token log-probabilities, teacher-forced, same masking as training). The pair is
`direction_correct` when `s_fwd > s_bwd`. Chance is 0.5. It asks: shown both versions, does the
model know which way the fix goes? It needs no extra training and is comparable across zero-shot,
few-shot and LoRA. Reuse the masking/scoring pattern of `llm_utils.predict_buggy`.

### 6.3 Trivial reference points to print in every report

- copy baseline: exact match 0 by construction, BLEU as above;
- chance for direction: 0.5;
- chance for localization: computed empirically as the expected overlap of one random line edit.

### 6.4 Uncertainty

`experiment_utils.bootstrap_ci(values, n=2000, seed=42)` returns the mean and a 95% percentile
interval over pairs. Report it for exact match, localization and direction. For the headline
comparison (LoRA vs zero-shot, and LoRA vs 3-shot) also report a paired bootstrap interval of the
difference on the same test pairs. `make_report.py` does this from the prediction files.

### 6.5 Protocol for val and test

The server window is one-shot and someone else may be running it, so every run writes predictions
for both val and test. The discipline is enforced in reporting:

- All comparisons and ablation tables (rank, quantization, model size) are reported on **val**.
- `make_report.py` picks the final configuration by val exact match, writes that choice to
  `results/selection.json` (with the val table it was based on), and only then reads test.
- The headline table shows test numbers for: zero-shot, 3-shot, and the selected LoRA model
  (plus 33B zero-shot if it ran). Test numbers of non-selected runs go to an appendix table clearly
  marked as not used for selection.
- Say this plainly in the report.

### 6.6 `metrics.json` contents

Metrics and breakdowns; all CLI args; model id and revision; adapter path; git commit hash; package
versions; GPU name; wall-clock times (train, eval); examples/second; number of trainable and total
parameters; number of examples dropped for length; training log history.

---

## 7. Legacy detection track (WP4)

Keep `bilstm.py`, `llm_baseline.py`, `train_lora.py` working; they answer research question 7.

Minimal changes only:

- `--data-dir` default to the small tier; accept the full tier.
- `llm_utils.load_model`: add a dtype option defaulting to bf16 on CUDA when supported; keep 4-bit.
- `train_lora.py`: refuse to run with more than one visible GPU (same message as 5.4); add
  `--time-budget-min`; allow `--model` instruct variants. The classification prompt and scoring
  stay as they are so zero-shot and LoRA remain comparable.

---

## 8. Experiment matrix

Run names are fixed; `make_report.py` relies on them.

| Run | Model | Method | Task | Train pairs | Priority |
|---|---|---|---|---|---|
| R1 `zs_1.3b` | 1.3B | zero-shot | repair | – | must |
| R2 `zs_6.7b` | 6.7B | zero-shot | repair | – | must |
| R3 `fs3_6.7b` | 6.7B | 3-shot | repair | – | must |
| R4 `lora_6.7b_r16` | 6.7B | LoRA r16 bf16 | repair | 50k | must (main) |
| R5 `lora_6.7b_r8` | 6.7B | LoRA r8 | repair | 50k | must |
| R6 `lora_6.7b_r32` | 6.7B | LoRA r32 | repair | 50k | must |
| R7 `lora_1.3b_r16` | 1.3B | LoRA r16 | repair | 50k | must |
| R8 `qlora_6.7b_r16` | 6.7B | QLoRA 4-bit r16 | repair | 50k | must |
| R9 `cls_zs_6.7b` | 6.7B | zero-shot | isolated detection | – | should |
| R10 `cls_lora_6.7b_r16` | 6.7B | LoRA r16 | isolated detection | 20k | should |
| R11 `zs_33b` | 33B, 4-bit, 2 GPUs | zero-shot | repair | – | optional |
| R12 `lora_6.7b_r16_s43`, `_s44` | 6.7B | main config, seeds 43/44 | repair | 50k | optional |

A local "insurance" version of R1 and R7 on the **small tier** with the 1.3B model
(`zs_1.3b_small`, `lora_1.3b_r16_small`) must be completed on a student GPU or Colab **before** the
server window. If the window is cancelled, the project still has a real before/after result.

---

## 9. Server (WP5)

### 9.1 Facts and unknowns

Known: AMD Threadripper Pro 7965X, 256 GB RAM, 6 × RTX 4090 (24 GB each), reached over VPN; the
instructor offered a window of about 3–4 hours and may launch the job himself; his own DeepSeek
service also runs on that machine.

Unknown, must be asked before the window (put these questions at the top of
`docs/SERVER_RUNBOOK.md`): OS and Python/CUDA driver versions; whether we get SSH or he runs it;
internet access from the server; free disk space; whether all six GPUs will be free during the
window or some are held by his service; whether we may create a venv.

Design for the worst case: no internet, some GPUs busy, a person who has never seen the repo runs
one command.

### 9.2 Principles

- Six cards are six separate 24 GB devices. One process per card, selected with
  `CUDA_VISIBLE_DEVICES`. No DataParallel, no DDP.
- 6.7B in bf16 is about 14 GB of weights and trains with LoRA on one card. 33B does not fit one card
  for training; it is evaluated zero-shot only, 4-bit, across two cards, and only if time remains.
- Everything is prepared offline: weights in the HF cache, parquet files copied, `HF_HUB_OFFLINE=1`.
- The window is run inside `tmux`; every process logs to `logs/<run>.log`; every run is idempotent,
  so re-running the same command after a crash continues instead of restarting.

### 9.3 `scripts/download_models.py`

`snapshot_download` for the model ids in 5.2 (33B behind a flag), prints resolved revisions and
cache sizes. Document the alternative when the server has no internet: download on another machine
and copy the HF cache directory.

### 9.4 `scripts/preflight.py`

Exits non-zero with a readable list of problems. Checks: Python and package versions; CUDA
available; for each GPU in `--gpus`: name, total and **free** memory (fail if under 22 GB free);
free disk (fail under 60 GB); data files of the requested tier exist with expected row counts;
each required model resolves from the local cache in offline mode; a real forward + backward pass
of the 1.3B model with LoRA on one GPU; bf16 support. Then a **timing run**: 30 optimizer steps of
the main config on one GPU, printing examples/second and the projected time for 50k pairs × 2
epochs, and the largest per-device batch size that fits (try 32, 16, 8).

### 9.5 `scripts/run_server.sh`

Environment variables with defaults: `GPUS="0 1 2 3 4 5"`, `BUDGET_MIN=150` (training budget per
run), `DATA=data/processed/full`, `BATCH`, `ACCUM`.

One background queue per GPU, each running its commands in order, then `wait`:

| GPU | Queue |
|---|---|
| 0 | R4 |
| 1 | R5 |
| 2 | R6 |
| 3 | R7, then R12 seed 43 if time |
| 4 | R8 |
| 5 | R1, R2, R3, R9, R10 |
| after all queues finish | R11 on two free GPUs, only if `RUN_33B=1` |

- If fewer GPUs are given, queues are merged in priority order (must → should → optional) rather
  than failing.
- The script starts with `preflight.py` and refuses to continue if it fails.
- A failing run must not stop the other queues; its queue moves to the next command, and the
  summary at the end lists which runs have `metrics.json` and which do not.
- Ends by running `make_report.py` and printing the val table.
- `SMOKE=1` runs the identical plan with the 1.3B model for everything, `--limit-pairs 64`,
  budget 2 minutes. **This rehearsal must pass on the real server (or at least on a student GPU)
  before the window.**

### 9.6 `docs/SERVER_RUNBOOK.md`

One page, numbered steps, copy-paste commands: questions for the instructor; what to copy to the
server; create venv and install; download or copy weights; `SMOKE=1 bash scripts/run_server.sh`;
the real run inside tmux; how to watch (`tail -f logs/*.log`, `nvidia-smi`); what to do if a run
dies; what to copy back (`runs/` without `checkpoints/`, `results/`, `logs/`). English and Russian.

---

## 10. Results, report, demo (WP6, WP7)

### 10.1 `make_report.py`

Reads `runs/*/metrics.json` and prediction files. Produces:

- `results/results.csv`: one row per run and split with the metrics and intervals;
- `results/selection.json`: section 6.5;
- `results/by_pattern.csv`: exact match and localization per `sstub_pattern` for the headline runs;
- paired-bootstrap differences for the headline comparisons.

### 10.2 `notebooks/06_results.ipynb`

Only reads files produced above. Figures saved to `results/figures/`:

1. headline bar chart: exact match on test for zero-shot, 3-shot, LoRA (with intervals);
2. rank ablation (val); 3. model-size and quantization comparison (val);
4. exact match per bug type, zero-shot vs LoRA; 5. training/validation loss curve of the main run;
6. detection summary: TF-IDF, BiLSTM, LLM zero-shot, LLM LoRA accuracy, plus direction accuracy.

Then **error analysis**: 15 sampled test pairs in three groups (fixed correctly; right line, wrong
fix; wrong line or copied input), shown as buggy / developer fix / model output, with a one-line
note each. The notes are written by the students; leave clearly marked placeholders, do not write
their analysis for them from imagination.

Also a cost table: trainable parameters, share of total, adapter size on disk, training time,
examples/second, per run.

### 10.3 Demo (`app/`)

FastAPI, one endpoint `POST /repair {code}` returning `{fixed, changed_lines, diff, parse_ok,
seconds}`, and `GET /` serving `index.html` (a textarea, a button, a side-by-side diff with the
changed line highlighted). Model and adapter chosen by environment variables; default is the 1.3B
model with the small-tier adapter so it runs on a laptop GPU or, slowly, on CPU. Model is loaded
once at startup. Input length is capped. The page states that the output is a suggestion from a
model trained on single-statement fixes and may be wrong.

---

## 11. Work packages

Order and parallelism: WP0 first. Then WP1, WP2, WP4, WP5-scripts can proceed in parallel (they
touch different files). WP3 needs WP2. WP6 needs the output formats of WP2/WP3. WP7 needs WP3.
WP8 last. Each WP ends with: tests pass (`python -m pytest tests -q`), smoke path run, a short
entry in the PR/commit description of what was and was not run.

**WP0. Baseline check.** Create the branch, install requirements, run existing tests, confirm they
pass before touching anything. Record failures instead of fixing unrelated things silently.

**WP1. Data tiers.** Section 4. Files: `data_utils.py`, `01_data.ipynb`, `tests/test_data_utils.py`,
`results/summary.json`. Done when: nesting and leakage tests pass; on a machine with raw data the
notebook produces both tiers and the summary; `02`/`03` re-run on the small tier.

**WP2. Repair core.** Sections 5.1, 6.1–6.4. Files: `repair_utils.py`, `experiment_utils.py`,
`tests/test_repair_utils.py`. Done when unit tests cover: prompt/label masking; output parsing
(fence, no fence, indentation kept); each metric on hand-made cases including pure insertion, pure
deletion, multi-line change, copied input; bootstrap determinism; direction scoring on a constant
fake LM (follow the fake-model style already in `tests/test_neural.py`).

**WP3. Training and evaluation scripts.** Sections 5.3–5.5, 6.5–6.6. Files: `train_repair.py`,
`eval_repair.py`, `llm_utils.py`. Done when: an end-to-end smoke test trains a tiny random causal
LM with LoRA for a few steps on fixtures and writes valid `metrics.json` and prediction files;
idempotency, resume, time-budget stop and the multi-GPU refusal are each tested or demonstrated.
On a GPU machine: the two small-tier insurance runs of section 8 are executed for real.

**WP4. Legacy detection track.** Section 7.

**WP5. Server tooling.** Section 9. Done when `SMOKE=1 bash scripts/run_server.sh` completes on a
Linux machine with at least one GPU (use `GPUS="0"`) and the queue-merging logic is shown to work.

**WP6. Reporting.** Section 10.1–10.2, `tests/test_report.py` on fabricated-for-test fixture
directories kept under `tests/fixtures/` only.

**WP7. Demo.** Section 10.3.

**WP8. Docs.** README (layout, how to run, results table filled only from real runs), `roadmap.md`,
`PROJECT_GUIDE_RU.md`: new sections in plain Russian covering why the task changed, what repair,
exact match, localization and fix direction mean, how LoRA/QLoRA settings were chosen, the val/test
protocol, how to run each script, and a "questions we may get at the defense" list with short
answers. No jargon without a one-line explanation.

---

## 12. Definition of done for the project

- All "must" runs of section 8 have `metrics.json`, or the report says exactly which are missing.
- `results/results.csv`, `selection.json`, the six figures and the error-analysis table exist and
  come from real runs.
- README results table matches `results.csv`.
- The seven research questions of section 1 each have a paragraph in the report with the number
  that answers it, including the unflattering ones.
- Limitations are stated: single-statement bugs only; fragments without file context; exact match
  undercounts alternative correct fixes; Python 2 share; exact-duplicate-only deduplication; one
  seed unless R12 ran; 33B evaluated in 4-bit.
- A new person can reproduce the small-tier pipeline from the README alone.

---

## 13. Prompts for launching agents

**Main build (run in the repo, on a machine that has `data/raw` if possible):**

> Read `CLAUDE.md` and `docs/AGENT_GUIDE.md` completely. Implement work packages WP0 through WP8 in
> the order and parallelism described in section 11, using subagents for packages that can run in
> parallel. Follow the rules in section 2 strictly, especially: never fabricate results, and say
> plainly what you could not run. Work on a branch named `repair-pipeline`. After each work package,
> run the tests and the smoke path, commit, and give me a three-line status. At the end give me:
> what was built, what was actually executed with real data/GPU, what remains for me to run by hand
> with exact commands, and anything in the guide you had to deviate from and why.

**Independent review (fresh session, after the build):**

> Read `docs/AGENT_GUIDE.md`. Review branch `repair-pipeline` against it as a skeptical reviewer.
> Check in particular: label masking and end-of-turn token in training examples, left padding in
> generation, the metric definitions in section 6.1 against their tests, test-split discipline
> (section 6.5), idempotency and time budget of the training script, and that no number in README
> or docs lacks a `metrics.json` behind it. Report findings ranked by severity; do not fix anything
> until I confirm.

**After the server window:**

> Read `docs/AGENT_GUIDE.md` sections 6.5, 10 and 12. The `runs/` directory now contains the real
> server results. Run `make_report.py`, execute `notebooks/06_results.ipynb`, update the README
> results table and the Russian guide from the real numbers only, and list which research questions
> of section 1 are answered and with what number. Leave the error-analysis notes for us to write.
