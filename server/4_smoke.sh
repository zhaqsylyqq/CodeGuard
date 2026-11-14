#!/usr/bin/env bash
# Step 4. Rehearsal: the same plan as the real run, but with the small model and 64 examples per
# run. Writes to runs_smoke/, logs_smoke/, results_smoke/ only. Must end with OK for every run.
. "$(dirname "$0")/_common.sh"
use_venv
SMOKE=1 bash scripts/run_server.sh
STATUS=$?
if [ "$STATUS" -eq 0 ]; then
    say "Rehearsal passed. Next: bash server/5_run.sh"
else
    say "Rehearsal FAILED (exit $STATUS). Do not start the real run. Send logs_smoke/ to the students."
fi
exit "$STATUS"
