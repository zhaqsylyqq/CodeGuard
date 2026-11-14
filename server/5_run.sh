#!/usr/bin/env bash
# Step 5. The real run. Starts in the background (tmux), so closing the laptop or losing the VPN
# does not stop it. Running this again after a crash continues where it stopped.
. "$(dirname "$0")/_common.sh"
use_venv
mkdir -p logs
SESSION=codeguard

say "Plan"
PLAN_ONLY=1 bash scripts/run_server.sh | sed -n '1,/^Commands:/p' | sed '$d'

if command -v tmux >/dev/null; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        die "A run is already going. Watch it: tmux attach -t $SESSION   (status: bash server/6_status.sh)"
    fi
    tmux new-session -d -s "$SESSION" \
        "cd '$ROOT' && . server/_common.sh && use_venv && bash scripts/run_server.sh 2>&1 | tee -a logs/server_window.log; echo; echo FINISHED - press Enter; read _"
    say "Started in tmux session '$SESSION'."
    echo "Watch live:   tmux attach -t $SESSION      (leave without stopping: Ctrl-b, then d)"
else
    nohup bash scripts/run_server.sh >> logs/server_window.log 2>&1 &
    say "Started with nohup (pid $!)."
    echo "Watch live:   tail -f logs/server_window.log   (Ctrl-c stops watching, not the run)"
fi
echo "Status:       bash server/6_status.sh"
echo "Stop it all:  bash server/stop.sh"
