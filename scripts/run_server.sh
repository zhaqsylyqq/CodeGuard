#!/usr/bin/env bash
# The whole server window in one command (plan: scripts/server_plan.py; operator kit: server/).
#
#   PLAN_ONLY=1 bash scripts/run_server.sh                 # print the plan, run nothing
#   PLAN_ONLY=1 ASSUME_SPEED=3 ... bash scripts/run_server.sh  # plan with assumed speeds (marked ASSUMED)
#   SMOKE=1 GPUS="0" bash scripts/run_server.sh            # rehearsal: 1.3B everywhere, tiny prefixes
#   bash scripts/run_server.sh                             # the real window (inside tmux)
#
# Server plan (architect, 2026-11-14): only what Colab cannot do, no 1.3B runs.
#   GPU 0: zs_6.7b, fs3_6.7b, lora_6.7b_r16_4k | GPU 1: lora_6.7b_r16 on N pairs, 1 epoch
#   GPUs 2+3: zs_33b, fs3_33b | GPUs 4+5: qlora_33b_r16 | optional: lora_6.7b_r8, lora_6.7b_r32
# N = min(50000, 0.85 x measured examples/s x training budget), rounded down to 1000. The speeds come
# from preflight (logs/preflight_timing.json, written by server/3_check.sh); without that file the
# script runs the timing first. Every queue's worst case fits WINDOW_MIN - MARGIN_MIN.
#
# Each queue is one background process chain on its GPU(s) (CUDA_VISIBLE_DEVICES); a two-GPU run is
# one process with the 4-bit model split over both cards (no DataParallel, no DDP). A failed run does
# not stop its queue. Runs whose <runs>/<name>/metrics.json exists are skipped, and the python
# scripts resume from checkpoints, so after a crash run the same command again. Logs: <logs>/<run>.log.

set -u
cd "$(dirname "$0")/.." || exit 1

GPUS="${GPUS:-0 1 2 3 4 5}"          # physical GPU indices as nvidia-smi shows them
WINDOW_MIN="${WINDOW_MIN:-210}"      # the whole window, minutes
MARGIN_MIN="${MARGIN_MIN:-20}"       # every queue's worst case ends this much before the window does
WITH_33B="${WITH_33B:-${RUN_33B:-1}}"  # 1: 33B evaluation and training on two GPU pairs
DATA="${DATA:-data/processed/full}"
BATCH="${BATCH:-8}"                  # per-device train batch (6.7B); preflight recommends it
ACCUM="${ACCUM:-4}"                  # BATCH x ACCUM must be 32
EVAL_BATCH="${EVAL_BATCH:-16}"       # generation / scoring batch (33B runs use at most 8)
PYTHON="${PYTHON:-python}"
PLAN_PYTHON="${PLAN_PYTHON:-$PYTHON}"  # interpreter for the planning helper
SMOKE="${SMOKE:-0}"
PLAN_ONLY="${PLAN_ONLY:-0}"

for flag in SMOKE PLAN_ONLY WITH_33B; do
    case "${!flag}" in 0|1) ;; *) echo "$flag must be 0 or 1, got '${!flag}'" >&2; exit 2 ;; esac
done
if [ $((BATCH * ACCUM)) -ne 32 ]; then
    echo "warning: BATCH x ACCUM = $((BATCH * ACCUM)), the design uses 32" >&2
fi
if [ "$SMOKE" = "1" ]; then
    RUNS=runs_smoke; LOGS=logs_smoke; RESULTS=results_smoke
else
    RUNS=runs; LOGS=logs; RESULTS=results
fi
TIMING_JSON="$LOGS/preflight_timing.json"

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # CUDA numbers GPUs like nvidia-smi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1            # live logs for tail -f
export TOKENIZERS_PARALLELISM=false

read -r -a GPU_LIST <<< "${GPUS//,/ }"
N_GPUS=${#GPU_LIST[@]}
if [ "$N_GPUS" -lt 1 ]; then
    echo "GPUS is empty" >&2; exit 2
fi

M_1_3B="deepseek-ai/deepseek-coder-1.3b-instruct@e063262dac8366fc1f28a4da0ff3c50ea66259ca"
M_6_7B="deepseek-ai/deepseek-coder-6.7b-instruct@e5d64addd26a6a1db0f9b863abf6ee3141936807"
M_33B="deepseek-ai/deepseek-coder-33b-instruct@61dc97b922b13995e7f83b7c8397701dbf9cfd4c"

PLAN_ARGS=(--gpus "${GPU_LIST[*]}" --window-min "$WINDOW_MIN" --margin-min "$MARGIN_MIN" --with-33b "$WITH_33B"
           --smoke "$SMOKE" --timing-json "$TIMING_JSON" --batch "$BATCH" --accum "$ACCUM"
           --eval-batch "$EVAL_BATCH" --data "$DATA" --runs "$RUNS" --python "$PYTHON")
# Assumed speeds (only for planning before preflight has measured them; printed as ASSUMED).
for pair in "ASSUME_SPEED speed" "ASSUME_EVAL_SPEED eval-speed" "ASSUME_EVAL_SPEED_3SHOT eval-speed-3shot" \
            "ASSUME_33B_SPEED 33b-speed" "ASSUME_33B_EVAL_SPEED 33b-eval-speed" \
            "ASSUME_33B_EVAL_SPEED_3SHOT 33b-eval-speed-3shot"; do
    read -r var flag <<< "$pair"
    if [ -n "${!var:-}" ]; then PLAN_ARGS+=("--assume-$flag" "${!var}"); fi
done

if [ "$PLAN_ONLY" = "1" ]; then
    "$PLAN_PYTHON" scripts/server_plan.py "${PLAN_ARGS[@]}" --format text
    exit $?
fi

stamp() { date '+%H:%M:%S'; }
START_EPOCH=$(date +%s)
mkdir -p "$LOGS" "$RUNS"

# ---------------------------------------------------------------- preflight

PREFLIGHT=("$PYTHON" scripts/preflight.py --gpus "${GPU_LIST[*]}" --data-dir "$DATA" --check-model "$M_1_3B"
           --eval-batch "$EVAL_BATCH" --timing-json "$TIMING_JSON" --window-min "$WINDOW_MIN"
           --margin-min "$MARGIN_MIN")
PAIR_33B="${GPU_LIST[$((N_GPUS - 2))]:-} ${GPU_LIST[$((N_GPUS - 1))]}"  # the last two GPUs
if [ "$SMOKE" = "1" ]; then
    PREFLIGHT+=(--smoke --models "$M_1_3B" --timing-model "$M_1_3B")
    if [ "$WITH_33B" = "1" ] && [ "$N_GPUS" -ge 2 ]; then
        PREFLIGHT+=(--check-33b "$M_1_3B" --pair "$PAIR_33B")  # rehearses the two-GPU check
    fi
else
    MODELS=("$M_1_3B" "$M_6_7B")
    if [ "$WITH_33B" = "1" ]; then MODELS+=("$M_33B"); fi
    PREFLIGHT+=(--models "${MODELS[@]}")
    if "$PLAN_PYTHON" -c 'import json, sys; d = json.load(open(sys.argv[1]))
sys.exit(0 if "main" in d and (sys.argv[2] == "0" or "33b" in d) else 1)' "$TIMING_JSON" "$WITH_33B" 2>/dev/null; then
        echo "[$(stamp)] speeds already measured ($TIMING_JSON); preflight without the timing run"
        PREFLIGHT+=(--skip-timing)
    else
        echo "[$(stamp)] no measured speeds in $TIMING_JSON yet: preflight runs the timing now (window time)"
        if [ "$WITH_33B" = "1" ] && [ "$N_GPUS" -ge 2 ]; then
            PREFLIGHT+=(--check-33b "$M_33B" --pair "$PAIR_33B")
        fi
    fi
fi
printf '[%s] preflight:' "$(stamp)"; printf ' %q' "${PREFLIGHT[@]}"; echo
"${PREFLIGHT[@]}" 2>&1 | tee -a "$LOGS/preflight.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Preflight failed; nothing was started. Fix the problems above (see $LOGS/preflight.log)." >&2
    exit 1
fi

# ---------------------------------------------------------------- plan

echo
"$PLAN_PYTHON" scripts/server_plan.py "${PLAN_ARGS[@]}" --format text | tee "$LOGS/plan.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "Planning failed (see above)." >&2; exit 1; }
declare -a QUEUE_RUNS QUEUE_DEVICES QUEUE_LATE
SHELL_PLAN="$("$PLAN_PYTHON" scripts/server_plan.py "${PLAN_ARGS[@]}" --format shell)" || exit 1
eval "$SHELL_PLAN"

# ---------------------------------------------------------------- queues

time_left_for() {  # run: optional runs start only if their worst case still fits
    local elapsed=$((($(date +%s) - START_EPOCH) / 60))
    [ $((elapsed + ${RUN_WORST[$1]} + MARGIN_MIN)) -le "$WINDOW_MIN" ]
}

run_one() {  # run
    local run="$1" log="$LOGS/$1.log" devices="${RUN_DEVICES[$1]}" status
    if [ -f "$RUNS/$run/metrics.json" ]; then
        echo "[$(stamp)] $run: $RUNS/$run/metrics.json exists, skipping"
        return 0
    fi
    if [ "${RUN_OPTIONAL[$run]}" = "1" ] && ! time_left_for "$run"; then
        echo "[$(stamp)] $run: optional and its worst case (${RUN_WORST[$run]} min) no longer fits, skipping"
        return 0
    fi
    echo "[$(stamp)] GPU $devices: start $run, log $log"
    {
        echo "=== $(date) CUDA_VISIBLE_DEVICES=$devices"
        echo "${RUN_CMD[$run]}"
    } >> "$log"
    eval "CUDA_VISIBLE_DEVICES=$devices ${RUN_CMD[$run]}" >> "$log" 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "[$(stamp)] GPU $devices: done  $run"
    else
        echo "[$(stamp)] GPU $devices: FAILED $run (exit $status), see $log; queue continues"
    fi
    return 0
}

run_queue() {  # queue index
    local run
    for run in ${QUEUE_RUNS[$1]}; do
        run_one "$run"
    done
    echo "[$(stamp)] GPU ${QUEUE_DEVICES[$1]}: queue finished"
}

# Ctrl-C / kill: background queues ignore SIGINT in a script, so stop the whole process group.
trap 'trap - INT TERM; echo "[$(stamp)] interrupted, stopping all queues"; kill 0; exit 130' INT TERM

echo
for ((q = 0; q < QUEUE_COUNT; q++)); do
    if [ "${QUEUE_LATE[$q]}" = "0" ]; then run_queue "$q" & fi
done
wait
for ((q = 0; q < QUEUE_COUNT; q++)); do  # smoke only: the two-GPU rehearsal after the others
    if [ "${QUEUE_LATE[$q]}" = "1" ]; then run_queue "$q"; fi
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
    for run in ${QUEUE_RUNS[$q]}; do
        kind=$([ "${RUN_OPTIONAL[$run]}" = "1" ] && echo optional || echo planned)
        if [ -f "$RUNS/$run/metrics.json" ]; then
            printf '  OK       %-20s %s\n' "$run" "$kind"
        else
            printf '  MISSING  %-20s %s  (see %s)\n' "$run" "$kind" "$LOGS/$run.log"
            if [ "$kind" = planned ]; then missing_required=$((missing_required + 1)); fi
        fi
    done
done
for run in $DROPPED; do
    printf '  DROPPED  %-20s not planned (see %s)\n' "$run" "$LOGS/plan.txt"
done
echo "Total time: $((($(date +%s) - START_EPOCH) / 60)) min."
if [ "$missing_required" -gt 0 ]; then
    echo "$missing_required planned run(s) have no metrics.json. Re-running this same command resumes them."
    exit 1
fi
