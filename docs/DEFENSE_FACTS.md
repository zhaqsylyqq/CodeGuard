# Defense facts

Every setting the runs used and where it is defined. Numbers come from the runs' `metrics.json` (A100
runs: `out_server/night_20261117/runs/`, Colab run: `runs/lora_1.3b_r16_small/`) or from the data
files. File references are `file:line` in this repository.

## 1. Task and data

| Fact | Value | Where |
|---|---|---|
| Task | Given a buggy Python fragment (one wrong statement), generate the whole fixed fragment | `src/repair_utils.py:15-19` (instruction) |
| Source | CTSSB-1M (cleaned TSSB-3M), single-statement bug fixes mined from GitHub | README |
| Raw rows | 966,985 | `data/processed/pool.json` |
| Candidate pool the tiers are sampled from | 714,241 pairs from 134,087 projects | `data/processed/pool.parquet` (built in `notebooks/01_data.ipynb`) |
| Length limit | at most 2000 characters and 60 lines per snippet | `src/data_utils.py:267-268`, `:275` |
| Split | by project (one project lives in one split only), 80/10/10, seed 42; cross-split duplicates removed | `src/data_utils.py:165`, `:198` |
| Small tier | train 4000 / val 500 / test 500 pairs (3390 / 429 / 413 projects) | `data/processed/small/` |
| Full tier | train 50,000 / val 2000 / test 5000 pairs (25,902 / 1376 / 2653 projects) | `data/processed/full/` |
| Nesting | the small tier is a prefix of the full tier, so "first N pairs" is the same set in both | `src/experiment_utils.py:34` (`load_pairs(first=N)`) |
| Evaluated pairs | small-tier runs: 500 val / 500 test; 33B runs: 150 val / 500 test; `zs_6.7b`, `fs3_6.7b`, `lora_6.7b_r16_50k`: 500 val / 2000 test (first 2000 full-tier test pairs; the first 500 are the small-tier test) | each run's `metrics.json` |

## 2. Models

| Model | Revision (commit) | Total parameters (from `metrics.json`) |
|---|---|---|
| deepseek-ai/deepseek-coder-1.3b-instruct | e063262dac8366fc1f28a4da0ff3c50ea66259ca | 1,346,471,936 |
| deepseek-ai/deepseek-coder-6.7b-instruct | e5d64addd26a6a1db0f9b863abf6ee3141936807 | 6,740,512,768 |
| deepseek-ai/deepseek-coder-33b-instruct | 61dc97b922b13995e7f83b7c8397701dbf9cfd4c | 33,342,991,360 (4-bit weights counted at full size) |

Pinned in `scripts/server_plan.py` (`MODELS`) and `scripts/download_models.py` (`PINNED`).

## 3. Training hyperparameters (repair, `src/train_repair.py`)

| Setting | Value | Where |
|---|---|---|
| Optimizer | AdamW, the Trainer default; in transformers 5.19.0 that is `adamw_torch_fused` (betas 0.9/0.999, eps 1e-8), checked against the installed version | not set in code; `TrainingArguments` at `src/train_repair.py:93-104` |
| Learning rate | 2e-4 | `src/train_repair.py:223`, used at `:96` |
| Schedule | cosine decay | `src/train_repair.py:97` |
| Warmup | 3% of the steps (`warmup_steps=0.03`; below 1 it is a ratio) | `src/train_repair.py:97` |
| Weight decay | 0.0 | `src/train_repair.py:97` |
| Gradient clipping | max norm 1.0 (Trainer default) | default |
| Effective batch | 32 pairs per optimizer step in every run | per run: see the table in section 5 |
| Batch × accumulation | 6.7B: 32 × 1; 33B: 8 × 4; 1.3B on the A100: 16 × 2; 1.3B Colab run: 8 × 4 | `scripts/a100_night.py:111-115` (6.7B), `:168-172` (33B); each run's `args` in `metrics.json` |
| Epochs | 1 (A100 runs); 2 (Colab run `lora_1.3b_r16_small`); default in code is 2 | `src/train_repair.py:219`; `scripts/a100_night.py:113` |
| Max length | 768 tokens; longer examples are dropped, never truncated (for example 97 of 50,000 in the 50k run) | `src/train_repair.py:224`, `:52-61` |
| Loss | cross-entropy on the answer tokens only; prompt tokens get label -100 | `src/repair_utils.py:90-94` |
| Batching | `group_by_length` sampling, padding per batch | `src/train_repair.py:101`, `src/llm_utils.py:152` |
| LoRA rank r | 8, 16 (main), 32 | `src/train_repair.py:214` |
| LoRA alpha | 2 × r (16, 32, 64) | `src/train_repair.py:44` |
| LoRA dropout | 0.05 | `src/train_repair.py:44` |
| Target modules | every linear layer in the decoder blocks: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (not the LM head) | `src/train_repair.py:18-27`, `:45` |
| Precision | bf16 on the A100 (Ampere or newer); fp16 on older GPUs; fp32 on CPU | `src/llm_utils.py:48-55`, `src/train_repair.py:100-101` |
| QLoRA | 4-bit NF4 base weights, double quantization, bf16 compute; `prepare_model_for_kbit_training` | `src/llm_utils.py:86-88`, `src/train_repair.py:42-43` |
| Gradient checkpointing | on (`use_reentrant=False`) | `src/train_repair.py:47` |
| Seed | 42 (seed replicates: 43, 44) | `src/train_repair.py:232`, `src/experiment_utils.py:21` |
| Time budget | stops training cleanly if exceeded; no run in the tables hit it (`stopped_by_time_budget: false`) | `src/llm_utils.py:116-143` |

## 4. Checkpoint and model selection (no test data involved)

1. **Within a run.** The model is evaluated on held-out val loss four times per epoch, using the first 500
   val pairs (150 for 33B). The checkpoint with the lowest val loss is kept as the adapter
   (`load_best_model_at_end`, `metric_for_best_model="eval_loss"`).
   Code: `src/train_repair.py:78-79`, `:90`, `:98-99`, `:126`.
2. **Across runs.** `src/make_report.py` picks the final run before it reads any test number:
   - it takes the highest val exact match among non-replicate LoRA/QLoRA runs of the full tier;
   - the candidates are compared on the val pairs all of them predicted (150);
   - ties go to fewer trainable parameters, then to the run name.

   Code: `src/make_report.py:139`. The result is written to `results/selection.json`:
   `lora_6.7b_r16_50k` and `qlora_33b_r16_25k` tie at val exact match 0.233 on the 150 shared pairs, and
   the 6.7B run wins on fewer trainable parameters.

## 5. Cost per run (from `metrics.json`)

Training minutes are training only, without model loading or evaluation. Examples/second = examples
seen / training seconds.

| Run | Trainable parameters | Share of total | Adapter MB | Train minutes | Examples/s | Hardware |
|---|---|---|---|---|---|---|
| lora_1.3b_r16_4k | 14,991,360 | 1.10% | 59.4 | 6.1 | 10.87 | A100-PCIE-40GB |
| lora_1.3b_r16_small (2 epochs) | 14,991,360 | 1.10% | 59.4 | 35.1 | 3.79 | Colab T4, fp16 (training); re-evaluated with the current generation rule on an RTX 3050 Laptop 4 GB |
| lora_6.7b_r8_4k | 19,988,480 | 0.30% | 78.5 | 9.6 | 6.90 | A100 |
| lora_6.7b_r16_4k (seeds 42/43/44) | 39,976,960 | 0.59% | 154.7 | 9.7 / 9.7 / 9.6 | 6.88 / 6.89 / 6.91 | A100 |
| lora_6.7b_r32_4k | 79,953,920 | 1.17% | 307.2 | 9.7 | 6.85 | A100 |
| qlora_6.7b_r16_4k | 39,976,960 | 0.59% | 154.7 | 11.4 | 5.83 | A100 |
| lora_6.7b_r16 (8000 pairs) | 39,976,960 | 0.59% | 154.7 | 17.3 | 7.70 | A100 |
| lora_6.7b_r16_50k | 39,976,960 | 0.59% | 154.7 | 98.2 | 8.47 | A100 |
| qlora_33b_r16_4k | 123,166,720 | 0.37% | 472.1 | 40.2 | 1.66 | A100 |
| qlora_33b_r16_25k | 123,166,720 | 0.37% | 472.1 | 237.8 | 1.75 | A100 |

Share = trainable / total parameters as stored in the run (total includes the adapter). The full table
is in `results/costs.csv`.

## 6. Generation and metrics

| Setting | Value | Where |
|---|---|---|
| Prompt | chat template of the instruct model; the user turn holds the instruction and the buggy code; 3-shot adds three train pairs as earlier turns | `src/repair_utils.py:31`, `:38` |
| Prefill | the assistant turn starts with ```` ```python\n ````, so the model writes code at once (rule `prefill_fence_v2`) | `src/repair_utils.py:20-22` |
| Decoding | greedy (`do_sample=False`); stops at the closing fence or at `min(512, 1.5 × input tokens + 32)` new tokens | `src/repair_utils.py:123-125`, `:314` |
| exact_match | the normalized prediction equals the developer's fixed fragment | `src/repair_utils.py:206` |
| localized | the model changed something, and at least one changed line is the developer's line | `src/repair_utils.py:208` |
| localized_exact | the model changed exactly the developer's lines, no more and no fewer | `src/repair_utils.py:209` |
| direction_correct | the model gives the fixed fragment a higher log-probability than the buggy one | `src/repair_utils.py` (scores `s_fwd`, `s_bwd`) |
| Intervals | 95% percentile bootstrap, 2000 resamples, seed 42; paired differences resample the pairs shared by both runs | `src/experiment_utils.py:70`, `:81` |
| Comparisons | only on pair_ids all compared runs predicted, with the pair count shown | `src/make_report.py:193`, `:312`; `scripts/compare_shared.py` |

## 7. Detection track (legacy, `src/train_lora.py`, `src/llm_baseline.py`)

| Setting | Value | Where |
|---|---|---|
| LoRA classifier | r 16, alpha 32, dropout 0.05, targets q/k/v/o projections, lr 2e-4, 1 epoch, batch 16 × 2, max length 512 | `src/train_lora.py:33-34`, `:80-87`; `scripts/a100_night.py:152-154` |
| Zero-shot | sum of answer-token log-probabilities, softmax over "fixed"/"buggy" | `src/llm_utils.py:166` |
| Evaluated on | small-tier val, 1000 snippets (500 buggy + 500 fixed) | `metrics.json` of `cls_*` |

## 8. Hardware and software

| Item | Value |
|---|---|
| Server GPU | 1 × NVIDIA A100-PCIE-40GB (driver 550, CUDA 12.4); one training process at a time |
| Colab GPU | NVIDIA T4 16 GB, fp16 (training of `lora_1.3b_r16_small`) |
| Laptop GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB (demo and local evaluation) |
| Python | 3.12.16 (server), 3.12.10 (laptop) |
| torch | 2.14.1+cu126 |
| transformers | 5.19.0 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| pandas / numpy | 3.0.6 / 2.5.3 |
| Demo | fastapi 0.162.5, uvicorn 0.54.0 |

Versions are taken from `environment` in the runs' `metrics.json` (server) and from the laptop `.venv`.
