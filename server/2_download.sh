#!/usr/bin/env bash
# Step 2. Download the model weights at the pinned revisions. Needs internet (huggingface.co).
# Do this BEFORE the window: 1.3B is about 3 GB, 6.7B about 14 GB, 33B about 67 GB.
# Safe to run again (resumes). DRY_RUN=1 lists what would be downloaded and downloads nothing.
. "$(dirname "$0")/_common.sh"
use_venv
export HF_HUB_OFFLINE=0

MODELS=("$(model_spec 1.3b)" "$(model_spec 6.7b)")
if [ "${WITH_33B:-0}" = "1" ]; then MODELS+=("$(model_spec 33b)"); fi
EXTRA=()
if [ "${DRY_RUN:-0}" = "1" ]; then EXTRA+=(--dry-run); fi
say "Downloading weights: ${MODELS[*]}"
python scripts/download_models.py --models "${MODELS[@]}" "${EXTRA[@]}" \
    || die "Download failed. Run this script again; it resumes."
say "Step 2 done. Next: bash server/3_check.sh"
