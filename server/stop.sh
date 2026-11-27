#!/usr/bin/env bash
# Emergency stop: ends the run (and a running preparation) and frees the GPUs. Finished runs are
# kept; 5_run.sh (or run_all.sh) continues later.
. "$(dirname "$0")/_common.sh"
tmux kill-session -t codeguard 2>/dev/null && echo "tmux session codeguard stopped."
tmux kill-session -t codeguard-prep 2>/dev/null && echo "tmux session codeguard-prep stopped."
pkill -f "server/run_all.sh" 2>/dev/null
pkill -f "scripts/run_server.sh" 2>/dev/null
pkill -f "src/(train_repair|eval_repair|train_lora|llm_baseline).py" 2>/dev/null
pkill -f "scripts/preflight.py" 2>/dev/null
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv
