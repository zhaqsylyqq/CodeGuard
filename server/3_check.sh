#!/usr/bin/env bash
# Step 3. Check everything and measure the speed on this server: 6.7B training and evaluation on
# the first GPU, and (WITH_33B=1) the 33B model in 4-bit over the last two GPUs. Takes about 10-30
# minutes. Do this before the window: the measured speeds go to logs/preflight_timing.json and the
# real run plans from them without spending window time.
. "$(dirname "$0")/_common.sh"
use_venv
mkdir -p logs

say "GPUs right now"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

read -r -a GPU_LIST <<< "${GPUS//,/ }"
MODELS=("$(model_spec 1.3b)" "$(model_spec 6.7b)")
ARGS=(--gpus "$GPUS" --data-dir data/processed/full --check-model "$(model_spec 1.3b)"
      --timing-model "$(model_spec 6.7b)" --eval-batch "$EVAL_BATCH" --timing-json logs/preflight_timing.json
      --window-min "$WINDOW_MIN")
if [ "${WITH_33B:-0}" = "1" ]; then
    MODELS+=("$(model_spec 33b)")
    if [ "${#GPU_LIST[@]}" -ge 2 ]; then
        ARGS+=(--check-33b "$(model_spec 33b)" --pair "${GPU_LIST[-2]} ${GPU_LIST[-1]}")
    else
        echo "note: only one GPU in GPUS, the 33B check needs two; set WITH_33B=0"
    fi
fi
rm -f logs/preflight_timing.json  # measure afresh

say "Preflight with timing run (log: logs/preflight_server.log)"
python scripts/preflight.py "${ARGS[@]}" --models "${MODELS[@]}" 2>&1 | tee logs/preflight_server.log
[ "${PIPESTATUS[0]}" -eq 0 ] || die "Preflight failed. Read the list of problems above; nothing else will work until it passes."

say "The plan for the window, from these measurements"
PLAN_ONLY=1 bash scripts/run_server.sh | sed -n '1,/^Commands:/p' | sed '$d'

say "Step 3 done."
echo "Copy the 'recommended: BATCH=.. ACCUM=..' numbers into server/config.env."
echo "If a line above says '33B check FAILED' or '33B is TOO SLOW', set WITH_33B=0 in server/config.env."
echo "Send logs/preflight_server.log and logs/preflight_timing.json to the students. Next: bash server/4_smoke.sh"
