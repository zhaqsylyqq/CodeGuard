#!/usr/bin/env bash
# The whole server window in one command (docs/AGENT_GUIDE.md section 9.5, docs/SERVER_RUNBOOK.md).
#
#   PLAN_ONLY=1 GPUS="0 1" bash scripts/run_server.sh    # print queues and commands, run nothing
#   SMOKE=1 GPUS="0" bash scripts/run_server.sh          # rehearsal: 1.3B everywhere, 64 pairs, 2 min
#   BATCH=8 ACCUM=4 bash scripts/run_server.sh           # the real window (inside tmux)
#
# One background queue per GPU (CUDA_VISIBLE_DEVICES = that one GPU), each running its runs in
# order; a failed run does not stop its queue. Runs whose <runs>/<name>/metrics.json exists are
# skipped, and the python scripts resume from checkpoints, so after a crash just run the same
# command again. Every run logs to <logs>/<name>.log (appended, so earlier attempts are kept).

set -u
cd "$(dirname "$0")/.." || exit 1

GPUS="${GPUS:-0 1 2 3 4 5}"          # physical GPU indices as nvidia-smi shows them
BUDGET_MIN="${BUDGET_MIN:-150}"      # training minutes per run (the script then saves and evaluates)
DATA="${DATA:-data/processed/full}"
BATCH="${BATCH:-8}"                  # per-device train batch; preflight's timing run recommends it
ACCUM="${ACCUM:-4}"                  # BATCH x ACCUM must be 32
EVAL_BATCH="${EVAL_BATCH:-16}"       # generation / scoring batch
PYTHON="${PYTHON:-python}"
RUN_33B="${RUN_33B:-0}"
SMOKE="${SMOKE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"
WINDOW_MIN="${WINDOW_MIN:-240}"      # length of the window; only used to decide optional runs
EVAL_MIN="${EVAL_MIN:-45}"           # minutes reserved for evaluation after a training run

# Pinned model revisions (commit shas resolved with scripts/download_models.py --revisions-only on
# 2026-11-12). Set a variable to an empty string to use the cache's "main" instead.
M_1_3B="deepseek-ai/deepseek-coder-1.3b-instruct"
M_6_7B="deepseek-ai/deepseek-coder-6.7b-instruct"
M_33B="deepseek-ai/deepseek-coder-33b-instruct"
REV_1_3B="${REV_1_3B-e063262dac8366fc1f28a4da0ff3c50ea66259ca}"
REV_6_7B="${REV_6_7B-e5d64addd26a6a1db0f9b863abf6ee3141936807}"
REV_33B="${REV_33B-61dc97b922b13995e7f83b7c8397701dbf9cfd4c}"

for flag in SMOKE PLAN_ONLY RUN_33B; do
    case "${!flag}" in 0|1) ;; *) echo "$flag must be 0 or 1, got '${!flag}'" >&2; exit 2 ;; esac
done
if [ $((BATCH * ACCUM)) -ne 32 ]; then
    echo "warning: BATCH x ACCUM = $((BATCH * ACCUM)), the design uses 32" >&2
fi

if [ "$SMOKE" = "1" ]; then
    BUDGET_MIN=2
    RUNS=runs_smoke; LOGS=logs_smoke; RESULTS=results_smoke
else
    RUNS=runs; LOGS=logs; RESULTS=results
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # CUDA numbers GPUs like nvidia-smi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1            # live logs for tail -f
export TOKENIZERS_PARALLELISM=false

read -r -a GPU_LIST <<< "${GPUS//,/ }"
N_GPUS=${#GPU_LIST[@]}
if [ "$N_GPUS" -lt 1 ]; then
    echo "GPUS is empty" >&2; exit 2
fi

# ---------------------------------------------------------------- runs (section 8)

# Priority order used when fewer than 6 GPUs are given: must (headline comparison first), then
# should, then optional.
PRIORITY="lora_6.7b_r16 zs_6.7b fs3_6.7b lora_1.3b_r16 zs_1.3b qlora_6.7b_r16 lora_6.7b_r8 lora_6.7b_r32
          cls_zs_6.7b cls_lora_6.7b_r16 lora_6.7b_r16_s43 lora_6.7b_r16_s44"

label_of() {
    case "$1" in
        zs_1.3b) echo R1 ;; zs_6.7b) echo R2 ;; fs3_6.7b) echo R3 ;; lora_6.7b_r16) echo R4 ;;
        lora_6.7b_r8) echo R5 ;; lora_6.7b_r32) echo R6 ;; lora_1.3b_r16) echo R7 ;;
        qlora_6.7b_r16) echo R8 ;; cls_zs_6.7b) echo R9 ;; cls_lora_6.7b_r16) echo R10 ;;
        zs_33b) echo R11 ;; lora_6.7b_r16_s4*) echo R12 ;; *) echo "?" ;;
    esac
}

tier_of() {
    case "$1" in
        cls_*) echo should ;;
        zs_33b|lora_6.7b_r16_s4*) echo optional ;;
        *) echo must ;;
    esac
}

is_training() {
    case "$1" in lora_*|qlora_*|cls_lora_*) return 0 ;; *) return 1 ;; esac
}

# set_model SIZE -> MODEL, REV (every model is the 1.3B one in smoke mode)
set_model() {
    local size="$1"
    if [ "$SMOKE" = "1" ]; then size=1.3b; fi
    case "$size" in
        1.3b) MODEL="$M_1_3B"; REV="$REV_1_3B" ;;
        6.7b) MODEL="$M_6_7B"; REV="$REV_6_7B" ;;
        33b) MODEL="$M_33B"; REV="$REV_33B" ;;
    esac
}

model_spec() {  # id@sha for preflight
    set_model "$1"
    if [ -n "$REV" ]; then echo "$MODEL@$REV"; else echo "$MODEL"; fi
}

# The helpers below fill the array CMD with the exact command of one run.
add_common() {  # $1 = --limit-pairs value for real runs ("" for none)
    if [ -n "$REV" ]; then CMD+=(--revision "$REV"); fi
    if [ "$SMOKE" = "1" ]; then
        CMD+=(--limit-pairs 64)
    elif [ -n "$1" ]; then
        CMD+=(--limit-pairs "$1")
    fi
}

eval_cmd() {  # run size shots [extra flags...]
    local run="$1" size="$2" shots="$3"; shift 3
    set_model "$size"
    CMD=("$PYTHON" src/eval_repair.py --model "$MODEL" --shots "$shots" "$@" --data-dir "$DATA"
         --splits val test --output "$RUNS/$run" --batch-size "$EVAL_BATCH" --seed 42)
    add_common ""
}

train_cmd() {  # run size rank seed [extra flags...]
    local run="$1" size="$2" rank="$3" seed="$4"; shift 4
    set_model "$size"
    CMD=("$PYTHON" src/train_repair.py --model "$MODEL" --data-dir "$DATA" --output "$RUNS/$run"
         --rank "$rank" "$@" --epochs 2 --batch-size "$BATCH" --accumulation-steps "$ACCUM"
         --eval-batch-size "$EVAL_BATCH" --time-budget-min "$BUDGET_MIN" --seed "$seed")
    add_common ""
}

build_cmd() {
    local run="$1"
    case "$run" in
        zs_1.3b) eval_cmd "$run" 1.3b 0 ;;
        zs_6.7b) eval_cmd "$run" 6.7b 0 ;;
        fs3_6.7b) eval_cmd "$run" 6.7b 3 ;;
        lora_6.7b_r16) train_cmd "$run" 6.7b 16 42 ;;
        lora_6.7b_r8) train_cmd "$run" 6.7b 8 42 ;;
        lora_6.7b_r32) train_cmd "$run" 6.7b 32 42 ;;
        lora_1.3b_r16) train_cmd "$run" 1.3b 16 42 ;;
        qlora_6.7b_r16) train_cmd "$run" 6.7b 16 42 --quantized ;;
        lora_6.7b_r16_s43) train_cmd "$run" 6.7b 16 43 ;;
        lora_6.7b_r16_s44) train_cmd "$run" 6.7b 16 44 ;;
        zs_33b) eval_cmd "$run" 33b 0 --quantized --multi-gpu ;;
        cls_zs_6.7b)
            set_model 6.7b
            CMD=("$PYTHON" src/llm_baseline.py --model "$MODEL" --data-dir "$DATA" --split val
                 --output "$RUNS/$run" --batch-size "$EVAL_BATCH")
            add_common "" ;;
        cls_lora_6.7b_r16)
            set_model 6.7b
            CMD=("$PYTHON" src/train_lora.py --model "$MODEL" --data-dir "$DATA" --output "$RUNS/$run"
                 --rank 16 --epochs 1 --batch-size "$BATCH" --accumulation-steps "$ACCUM"
                 --time-budget-min "$BUDGET_MIN")
            add_common 20000 ;;
        *) echo "unknown run $run" >&2; return 1 ;;
    esac
}

# ---------------------------------------------------------------- queues (section 9.5)

declare -a QUEUE
if [ "$N_GPUS" -ge 6 ]; then
    QUEUE[0]="lora_6.7b_r16"
    QUEUE[1]="lora_6.7b_r8"
    QUEUE[2]="lora_6.7b_r32"
    # GPU 3 trains the small 1.3B model and finishes early, so it also takes the detection runs
    # (R9, R10); the optional seed runs (R12) stay last. GPU 5 keeps the three repair evaluations.
    QUEUE[3]="lora_1.3b_r16 cls_zs_6.7b cls_lora_6.7b_r16 lora_6.7b_r16_s43 lora_6.7b_r16_s44"
    QUEUE[4]="qlora_6.7b_r16"
    QUEUE[5]="zs_1.3b zs_6.7b fs3_6.7b"
    QUEUE_COUNT=6
else
    QUEUE_COUNT=$N_GPUS
    i=0
    for run in $PRIORITY; do
        slot=$((i % N_GPUS))
        QUEUE[$slot]="${QUEUE[$slot]:-}${QUEUE[$slot]:+ }$run"
        i=$((i + 1))
    done
fi

LATE_RUNS=""
if [ "$RUN_33B" = "1" ]; then LATE_RUNS="zs_33b"; fi
if [ "$N_GPUS" -ge 2 ]; then
    GPUS_33B="${GPU_LIST[0]},${GPU_LIST[1]}"
else
    GPUS_33B="${GPU_LIST[0]}"
fi

stamp() { date '+%H:%M:%S'; }
START_EPOCH=$(date +%s)

print_plan() {
    echo "Plan: SMOKE=$SMOKE GPUS=\"${GPU_LIST[*]}\" BUDGET_MIN=$BUDGET_MIN DATA=$DATA BATCH=$BATCH ACCUM=$ACCUM" \
         "EVAL_BATCH=$EVAL_BATCH RUN_33B=$RUN_33B -> $RUNS/, $LOGS/"
    if [ "$N_GPUS" -gt 6 ]; then
        echo "note: only the first 6 GPUs get a queue; the 33B run uses the first two"
    fi
    local q run labels n_train
    for ((q = 0; q < QUEUE_COUNT; q++)); do
        labels=""; n_train=0
        for run in ${QUEUE[$q]}; do
            labels="$labels $(label_of "$run")"
            if is_training "$run" && [ "$(tier_of "$run")" != optional ]; then
                n_train=$((n_train + 1))
            fi
        done
        echo "GPU ${GPU_LIST[$q]} queue: ${QUEUE[$q]}"
        echo "      (${labels# }; $n_train must/should training run(s) x $BUDGET_MIN min budget + evaluation)"
        if [ $((n_train * BUDGET_MIN)) -gt "$WINDOW_MIN" ]; then
            echo "      WARNING: $((n_train * BUDGET_MIN)) min of training budget > WINDOW_MIN $WINDOW_MIN" \
                 "(worst case: every run uses its full budget); the later runs of this queue may not" \
                 "finish. Check the preflight timing, lower BUDGET_MIN, or give more GPUs."
        fi
    done
    if [ -n "$LATE_RUNS" ]; then
        echo "After all queues: $LATE_RUNS (R11) on GPU(s) $GPUS_33B"
    else
        echo "After all queues: zs_33b (R11) not run (RUN_33B=0)"
    fi
    echo "Optional runs (R12) start only if elapsed + BUDGET_MIN + EVAL_MIN ($EVAL_MIN) <= WINDOW_MIN ($WINDOW_MIN)."
    echo "Commands:"
    for ((q = 0; q < QUEUE_COUNT; q++)); do
        for run in ${QUEUE[$q]}; do
            build_cmd "$run"
            printf '  [GPU %s] %-18s CUDA_VISIBLE_DEVICES=%s' "${GPU_LIST[$q]}" "$run" "${GPU_LIST[$q]}"
            printf ' %q' "${CMD[@]}"; echo
        done
    done
    for run in $LATE_RUNS; do
        build_cmd "$run"
        printf '  [after]  %-18s CUDA_VISIBLE_DEVICES=%s' "$run" "$GPUS_33B"
        printf ' %q' "${CMD[@]}"; echo
    done
}

time_left_for_optional() {
    local elapsed=$((($(date +%s) - START_EPOCH) / 60))
    [ $((elapsed + BUDGET_MIN + EVAL_MIN)) -le "$WINDOW_MIN" ]
}

run_one() {  # devices run
    local devices="$1" run="$2" log="$LOGS/$2.log" status
    if [ -f "$RUNS/$run/metrics.json" ]; then
        echo "[$(stamp)] $run: $RUNS/$run/metrics.json exists, skipping"
        return 0
    fi
    if [ "$(tier_of "$run")" = optional ] && ! time_left_for_optional; then
        echo "[$(stamp)] $run: optional and not enough time left in the window, skipping"
        return 0
    fi
    build_cmd "$run" || return 1
    echo "[$(stamp)] GPU $devices: start $run ($(label_of "$run")), log $log"
    {
        echo "=== $(date) CUDA_VISIBLE_DEVICES=$devices"
        printf '%q ' "${CMD[@]}"; echo
    } >> "$log"
    CUDA_VISIBLE_DEVICES="$devices" "${CMD[@]}" >> "$log" 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "[$(stamp)] GPU $devices: done  $run"
    else
        echo "[$(stamp)] GPU $devices: FAILED $run (exit $status), see $log; queue continues"
    fi
    return 0
}

run_queue() {  # gpu runs...
    local gpu="$1" run; shift
    for run in "$@"; do
        run_one "$gpu" "$run"
    done
    echo "[$(stamp)] GPU $gpu: queue finished"
}

# ---------------------------------------------------------------- main

print_plan
if [ "$PLAN_ONLY" = "1" ]; then
    exit 0
fi

mkdir -p "$LOGS" "$RUNS"

PREFLIGHT=("$PYTHON" scripts/preflight.py --gpus "${GPU_LIST[*]}" --data-dir "$DATA"
           --check-model "$(model_spec 1.3b)")
if [ "$SMOKE" = "1" ]; then
    PREFLIGHT+=(--smoke --models "$(model_spec 1.3b)" --timing-model "$(model_spec 1.3b)")
else
    # The timing run is done once before the window (see the runbook); do not spend window time on it.
    PREFLIGHT+=(--skip-timing --models "$(model_spec 1.3b)" "$(model_spec 6.7b)")
    if [ "$RUN_33B" = "1" ]; then PREFLIGHT+=("$(model_spec 33b)"); fi
fi
echo
printf '[%s] preflight:' "$(stamp)"; printf ' %q' "${PREFLIGHT[@]}"; echo
"${PREFLIGHT[@]}" 2>&1 | tee "$LOGS/preflight.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Preflight failed; nothing was started. Fix the problems above (see $LOGS/preflight.log)." >&2
    exit 1
fi

# Ctrl-C / kill: background queues ignore SIGINT in a script, so stop the whole process group.
trap 'trap - INT TERM; echo "[$(stamp)] interrupted, stopping all queues"; kill 0; exit 130' INT TERM

echo
for ((q = 0; q < QUEUE_COUNT; q++)); do
    # shellcheck disable=SC2086  # run names have no spaces; word splitting is intended
    run_queue "${GPU_LIST[$q]}" ${QUEUE[$q]} &
done
wait

for run in $LATE_RUNS; do
    run_one "$GPUS_33B" "$run"
done

echo
echo "[$(stamp)] report: $PYTHON src/make_report.py --runs $RUNS --results $RESULTS"
"$PYTHON" src/make_report.py --runs "$RUNS" --results "$RESULTS" 2>&1 | tee "$LOGS/make_report.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "make_report.py failed (see $LOGS/make_report.log); the runs themselves are kept."
fi

echo
echo "Summary ($RUNS/<run>/metrics.json):"
missing_required=0
for ((q = 0; q < QUEUE_COUNT; q++)); do
    for run in ${QUEUE[$q]}; do
        ALL_RUNS="${ALL_RUNS:-} $run"
    done
done
for run in ${ALL_RUNS:-} $LATE_RUNS; do
    tier=$(tier_of "$run")
    if [ -f "$RUNS/$run/metrics.json" ]; then
        printf '  OK       %-4s %-20s %s\n' "$(label_of "$run")" "$run" "$tier"
    else
        printf '  MISSING  %-4s %-20s %s  (see %s)\n' "$(label_of "$run")" "$run" "$tier" "$LOGS/$run.log"
        if [ "$tier" != optional ]; then missing_required=$((missing_required + 1)); fi
    fi
done
echo "Total time: $((($(date +%s) - START_EPOCH) / 60)) min."
if [ "$missing_required" -gt 0 ]; then
    echo "$missing_required must/should run(s) have no metrics.json. Re-running this same command resumes them."
    exit 1
fi
