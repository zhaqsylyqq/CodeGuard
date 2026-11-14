#!/usr/bin/env bash
# Step 3. Check everything and measure the training speed on this server (about 5-10 minutes,
# uses the first GPU from config.env). Do this before the window if possible.
. "$(dirname "$0")/_common.sh"
use_venv
mkdir -p logs

say "GPUs right now"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

say "Preflight with timing run (log: logs/preflight_server.log)"
python scripts/preflight.py --gpus "$GPUS" --data-dir data/processed/full 2>&1 | tee logs/preflight_server.log
[ "${PIPESTATUS[0]}" -eq 0 ] || die "Preflight failed. Read the list of problems above; nothing else will work until it passes."

say "Step 3 done."
echo "Copy the 'recommended: BATCH=.. ACCUM=..' numbers into server/config.env."
echo "Send logs/preflight_server.log to the students. Next: bash server/4_smoke.sh"
