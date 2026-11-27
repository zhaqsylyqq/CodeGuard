#!/usr/bin/env bash
# Step 5. The real run. Starts in the background (tmux), so closing the laptop or losing the VPN
# does not stop it. Running this again after a crash continues where it stopped.
# DRY_RUN=1 prints the plan and the launch command and starts nothing.
. "$(dirname "$0")/_common.sh"
use_venv
mkdir -p logs
SESSION=codeguard
RUN="cd '$ROOT' && . server/_common.sh && use_venv && bash scripts/run_server.sh 2>&1 | tee -a logs/server_window.log; echo; echo FINISHED - press Enter; read _"

say "Plan"
PLAN="$(PLAN_ONLY=1 bash scripts/run_server.sh 2>&1)"
if [ $? -eq 0 ]; then
    printf '%s\n' "$PLAN" | sed -n '1,/^Commands:/p' | sed '$d'
else
    printf '%s\n' "$PLAN" | tail -3
    echo "(No plan yet: the speeds are not measured, so the run measures them first, in window time."
    echo " Better: stop with Ctrl-c now and run bash server/3_check.sh first.)"
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
    say "DRY_RUN=1: this would start now"
    if command -v tmux >/dev/null; then
        echo "tmux new-session -d -s $SESSION \"$RUN\""
    else
        echo "nohup bash scripts/run_server.sh >> logs/server_window.log 2>&1 &"
    fi
    exit 0
fi

if command -v tmux >/dev/null; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        die "A run is already going. Watch it: tmux attach -t $SESSION   (status: bash server/6_status.sh)"
    fi
    tmux new-session -d -s "$SESSION" "$RUN"
    say "Started in tmux session '$SESSION'."
    echo "Watch live:   tmux attach -t $SESSION      (leave without stopping: Ctrl-b, then d)"
else
    nohup bash scripts/run_server.sh >> logs/server_window.log 2>&1 &
    say "Started with nohup (pid $!)."
    echo "Watch live:   tail -f logs/server_window.log   (Ctrl-c stops watching, not the run)"
fi
echo "Status:       bash server/6_status.sh"
echo "Stop it all:  bash server/stop.sh"
