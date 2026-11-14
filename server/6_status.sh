#!/usr/bin/env bash
# Shows what is running and what is finished. Changes nothing; run it as often as you like.
. "$(dirname "$0")/_common.sh"
RUNS="${1:-runs}"; LOGS="${RUNS/runs/logs}"

say "GPUs"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

say "Runs in $RUNS/ (DONE = metrics.json written)"
found=0
for log in "$LOGS"/*.log; do
    [ -e "$log" ] || continue
    name="$(basename "$log" .log)"
    case "$name" in preflight*|make_report|server_window) continue ;; esac
    found=1
    if [ -f "$RUNS/$name/metrics.json" ]; then state="DONE   "; else state="running or failed"; fi
    last="$(tail -c 2000 "$log" | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -1 | cut -c1-110)"
    printf '%-22s %-18s %s\n' "$name" "$state" "$last"
done
[ "$found" = 1 ] || echo "(no runs started yet)"

if [ -f "$LOGS/server_window.log" ]; then
    say "Last lines of the main log"
    tail -5 "$LOGS/server_window.log"
fi
