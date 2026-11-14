#!/usr/bin/env bash
# Emergency stop: ends the run and frees the GPUs. Finished runs are kept; 5_run.sh continues later.
. "$(dirname "$0")/_common.sh"
tmux kill-session -t codeguard 2>/dev/null && echo "tmux session stopped."
pkill -f "scripts/run_server.sh" 2>/dev/null
pkill -f "src/(train_repair|eval_repair|train_lora|llm_baseline).py" 2>/dev/null
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv
