#!/usr/bin/env bash
# Everything in one command.
#
#   bash server/run_all.sh prepare   # steps 1-4: environment, weights, check, rehearsal (before the window)
#   bash server/run_all.sh run       # step 5: the real run (in the window)
#   bash server/run_all.sh all       # prepare, then run straight away
#
# It keeps going after the SSH connection drops (tmux), stops at the first step that fails, and
# skips steps that already passed, so after fixing a problem just run the same command again.
# To redo a step, delete its marker in .server_state/ (for example .server_state/3_check.done).
. "$(dirname "$0")/_common.sh"

MODE="${1:-}"
case "$MODE" in prepare|run|all) ;; *)
    echo "usage: bash server/run_all.sh prepare | run | all" >&2; exit 2 ;;
esac

# Re-start inside tmux so a dropped connection does not stop a long download or rehearsal.
if [ "$MODE" != run ] && [ -z "${TMUX:-}" ] && [ -z "${CODEGUARD_NO_TMUX:-}" ] && command -v tmux >/dev/null; then
    SESSION=codeguard-prep
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        die "Preparation is already going. Watch it: tmux attach -t $SESSION"
    fi
    tmux new-session -d -s "$SESSION" \
        "cd '$ROOT' && CODEGUARD_NO_TMUX=1 bash server/run_all.sh $MODE; echo; echo FINISHED - press Enter; read _"
    echo "Started in the background (tmux session '$SESSION')."
    echo "Watch live:  tmux attach -t $SESSION     (leave without stopping: Ctrl-b, then d)"
    echo "Or:          tail -f logs/run_all.log"
    exit 0
fi

mkdir -p logs .server_state
exec > >(tee -a logs/run_all.log) 2>&1
echo "=== $(date) run_all.sh $MODE"

step() {  # name script
    local name="$1" script="$2"
    if [ -f ".server_state/$name.done" ]; then
        echo "-- $name: already passed, skipping"
        return 0
    fi
    say "$name"
    if bash "server/$script"; then
        if [ "${DRY_RUN:-0}" = "1" ]; then echo "-- $name: DRY_RUN=1, not marked as passed"
        else date > ".server_state/$name.done"; fi
    else
        die "$name failed. Fix the problem printed above and run the same command again. Log: logs/run_all.log"
    fi
}

if [ "$MODE" != run ]; then
    step 1_setup 1_setup.sh
    step 2_download 2_download.sh
    step 3_check 3_check.sh
    step 4_smoke 4_smoke.sh
    say "Preparation finished: environment, weights, check and rehearsal all passed."
fi

if [ "$MODE" != prepare ]; then
    [ -f .server_state/4_smoke.done ] || die "The rehearsal has not passed yet. Run: bash server/run_all.sh prepare"
    bash server/5_run.sh
else
    echo "When the window starts: bash server/run_all.sh run"
fi
