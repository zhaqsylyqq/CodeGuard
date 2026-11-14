# Shared by the numbered scripts. Not meant to be run directly.
set -u
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$KIT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

# shellcheck disable=SC1091
. "$KIT_DIR/config.env"
export GPUS WINDOW_MIN BATCH ACCUM EVAL_BATCH
export RUN_33B="${WITH_33B:-0}"
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; else unset HF_HOME; fi

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

use_venv() {
    [ -f "$ROOT/.venv/bin/activate" ] || die "No .venv yet. Run: bash server/1_setup.sh"
    # shellcheck disable=SC1091
    . "$ROOT/.venv/bin/activate"
    export PYTHON=python
}
