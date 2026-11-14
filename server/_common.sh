# Shared by the numbered scripts. Not meant to be run directly.
set -u
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$KIT_DIR/.." && pwd)"
cd "$ROOT" || exit 1

# config.env edited on Windows may have CRLF line endings; strip them before reading.
# shellcheck disable=SC1090
. <(tr -d '\r' < "$KIT_DIR/config.env")
export GPUS WINDOW_MIN BATCH ACCUM EVAL_BATCH WITH_33B
if [ -n "${HF_HOME:-}" ]; then export HF_HOME; else unset HF_HOME; fi

say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

use_venv() {
    [ -f "$ROOT/.venv/bin/activate" ] || die "No .venv yet. Run: bash server/1_setup.sh"
    # shellcheck disable=SC1091
    . "$ROOT/.venv/bin/activate"
    export PYTHON=python
}

model_spec() {  # 1.3b | 6.7b | 33b -> id@sha, pinned in scripts/server_plan.py
    python -c "import sys; sys.path.insert(0, 'scripts'); from server_plan import MODELS; print('@'.join(MODELS['$1']))"
}
