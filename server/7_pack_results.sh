#!/usr/bin/env bash
# Step 7. Pack everything the students need into one file. Training checkpoints are left out
# (large and not needed); the final LoRA adapters are included.
. "$(dirname "$0")/_common.sh"
OUT="codeguard_results_$(date +%Y%m%d_%H%M).tar.gz"
DIRS=()
for d in runs logs results runs_smoke logs_smoke results_smoke; do
    [ -d "$d" ] && DIRS+=("$d")
done
[ "${#DIRS[@]}" -gt 0 ] || die "Nothing to pack yet."
nvidia-smi > logs/nvidia_smi_after.txt 2>&1 || true
tar --exclude='checkpoints' --exclude='__pycache__' -czf "$OUT" "${DIRS[@]}" || die "tar failed"
say "Packed: $ROOT/$OUT ($(du -h "$OUT" | cut -f1))"
echo "Copy it to the laptop (run this ON THE LAPTOP, not here):"
echo "  scp $(whoami)@<server-address>:$ROOT/$OUT ."
