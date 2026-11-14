#!/usr/bin/env bash
# Step 2. Download the model weights. Needs internet (huggingface.co). Do this BEFORE the window:
# 1.3B is about 3 GB, 6.7B about 14 GB, 33B about 67 GB. Safe to run again (resumes).
. "$(dirname "$0")/_common.sh"
use_venv
export HF_HUB_OFFLINE=0

ARGS=()
if [ "${WITH_33B:-0}" = "1" ]; then ARGS+=(--with-33b); fi
say "Downloading weights ${ARGS[*]:-}"
python scripts/download_models.py "${ARGS[@]}" || die "Download failed. Run this script again; it resumes."

say "Revisions (must match REV_* in scripts/run_server.sh)"
python scripts/download_models.py --revisions-only "${ARGS[@]}"
say "Step 2 done. Next: bash server/3_check.sh"
