#!/usr/bin/env bash
# Step 1. Create the Python environment and install the pinned packages. Needs internet (PyPI).
# Safe to run again.
. "$(dirname "$0")/_common.sh"

say "System"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver on this machine?"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
PY="${PY:-python3}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version; print("python", sys.version.split()[0])' \
    || die "Python 3.10 or newer is needed. Set PY=/path/to/python3.x and run again."
df -h "$ROOT" | tail -1

say "Virtual environment (.venv)"
if [ ! -d .venv ]; then
    "$PY" -m venv .venv || die "Could not create a venv (on Ubuntu: sudo apt install python3-venv)."
fi
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-neural.txt || die "pip install failed (no internet? see README, 'No internet')."

say "Check"
python - <<'PYEOF'
import torch, transformers, peft, bitsandbytes
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
print("CUDA available:", torch.cuda.is_available(), "| GPUs seen:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("torch does not see the GPUs: install the torch build matching the driver "
                     "(see the comment at the top of requirements-neural.txt)")
PYEOF
[ $? -eq 0 ] || exit 1
command -v tmux >/dev/null || echo "note: tmux is not installed; 5_run.sh will fall back to nohup."
say "Step 1 done. Next: bash server/2_download.sh"
