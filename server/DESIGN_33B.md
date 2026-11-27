# Design addendum: using the server for a model larger than 6.7B

Written by the architect on 2026-11-14, after the first real small-tier results. This extends
`docs/AGENT_GUIDE.md` sections 8 and 9; where they differ, this file wins. Nothing here has been
run: every number about 33B below is an estimate until `preflight` measures it on the server.

## Why

The server's value is what cannot be done on Colab. A 6.7B model in 4-bit already runs on a free
T4, so a plan that only trains 6.7B under-uses six 24 GB cards. The DeepSeek-Coder family has three
sizes (1.3B, 6.7B, 33B); the 33B one cannot be trained on any hardware the team has except this
server.

## Constraint

One RTX 4090 has 24 GB. 33B in 4-bit NF4 is roughly 19-20 GB of weights, which leaves too little
for training on one card. Across **two** cards (`device_map="auto"`, model split by layers, one
process) there is room. This is model parallelism inside one process, not distributed training:
still no DDP, no DataParallel. It is slow: expect several times slower per example than 6.7B.

## New and changed runs

| Run | Model | Method | Train pairs | GPUs | Priority |
|---|---|---|---|---|---|
| R11 `zs_33b` | 33B 4-bit | zero-shot | - | 2 | must (was optional) |
| R13 `qlora_33b_r16_4k` | 33B 4-bit | QLoRA r16 | first 4000 | 2 | must |
| R14 `lora_6.7b_r16_4k` | 6.7B bf16 | LoRA r16 | first 4000 | 1 | must |
| R8 `qlora_6.7b_r16` | 6.7B 4-bit | QLoRA r16 | N | 1 | should (was must) |
| new Colab run `qlora_1.3b_r16_small` | 1.3B 4-bit | QLoRA r16 | 4000 | T4 | must |
| new Colab run `cls_lora_1.3b_r16_small` | 1.3B | detection LoRA | 4000 | T4 | should |

"First 4000" is the small tier, which is a prefix of the full tier, with `--epochs 1`. Together with
the existing `lora_1.3b_r16_small` this gives a **model-size ladder at fixed data**:
1.3B / 6.7B / 33B, each trained on the same 4000 pairs and evaluated on the same pairs. That
answers research question 3 cleanly, and question 5 (tuned small model vs untuned large model) gets
a real 33B number. The quantization question (4) is answered on 1.3B in Colab, where it is cheap;
R8 repeats it on 6.7B only if time allows.

Caveat to state in the report: the existing `lora_1.3b_r16_small` trained for 2 epochs and kept the
best checkpoint by validation loss, which was the end of epoch 1; say so next to the ladder.

## Queues for six GPUs

| GPU | Queue (in order) |
|---|---|
| 0 | R4 `lora_6.7b_r16` |
| 1 | R5 `lora_6.7b_r8` |
| 2 | R6 `lora_6.7b_r32` |
| 3 | R2 `zs_6.7b`, R3 `fs3_6.7b`, R1 `zs_1.3b`, R14, R7 `lora_1.3b_r16`, then R9, R10 if time |
| 4+5 | R11 `zs_33b`, then R13 `qlora_33b_r16_4k`; after that, each card separately: R8, R12 if time |

Evaluation-only runs go first on GPU 3 because they are short, bounded and needed for the headline
table. With fewer than six free GPUs the priority order is: R4, R2, R3, R14, R11, R13, R1, R7, R5,
R6, then should, then optional. If fewer than two cards are free for the pair, the 33B runs are
skipped and reported as skipped.

## Requirements for the implementation

1. `train_repair.py`: add `--multi-gpu`. With it the single-GPU guard is lifted and the model loads
   with `device_map="auto"` over the visible cards; it requires `--quantized`. Verify with the
   installed `transformers`/`peft`/`accelerate` that `Trainer` then keeps the model as is (no
   DataParallel wrapping). Test the flag logic; test the real thing in preflight.
2. `preflight.py`: when 33B is requested, load it in 4-bit over two given GPUs, run 5 optimizer
   steps of QLoRA, print peak memory per card and examples/second, and print the projected time for
   4000 pairs. If it does not fit or the projection exceeds the time available, say so clearly and
   tell the operator to set `WITH_33B=0`. A failed 33B check must not fail the rest of preflight.
3. `run_server.sh`: the queues above; the pair of GPUs for 33B is the last two in `GPUS`; R13 gets
   a time budget equal to what remains of the window on those cards after R11, minus evaluation and
   the margin. If the budget stops R13 early, the report shows how many pairs it actually saw.
4. Evaluation subsets for the 33B runs: 500 val pairs and 1000 test pairs (prefixes, so they are
   subsets of what the other runs use). `make_report.py` compares on shared `pair_id`s.
5. SMOKE mode must exercise the two-GPU code path when two GPUs are given (with the 1.3B model),
   and skip it with a printed notice when only one is given.
6. The Colab notebook gets the two new small-tier runs listed above.
7. `server/` is the operator-facing kit (numbered scripts plus `README_RU.md`). Keep its scripts
   thin wrappers around `scripts/`; if a variable or behaviour of `run_server.sh` changes, update
   `server/config.env`, the kit scripts and `server/README_RU.md` so they stay true. Add a test or
   a `PLAN_ONLY` check that the kit runs end to end without a GPU up to the point of launching.

## What must be said honestly in the report

- 33B was trained on 4000 pairs, not 50,000, and in 4-bit; it is a size-scaling data point, not a
  tuned best model.
- All 33B speed and memory figures in this file were estimates before the run.
