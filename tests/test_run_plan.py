"""scripts/run_server.sh: queue plans (PLAN_ONLY=1) and the queue runner with a fake python.

Skipped where bash is not available. Nothing here loads a model.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_server.sh"
MUST = ["zs_1.3b", "zs_6.7b", "fs3_6.7b", "lora_6.7b_r16", "lora_6.7b_r8", "lora_6.7b_r32",
        "lora_1.3b_r16", "qlora_6.7b_r16"]
SHOULD = ["cls_zs_6.7b", "cls_lora_6.7b_r16"]
OPTIONAL = ["lora_6.7b_r16_s43", "lora_6.7b_r16_s44"]


# Full path: on Windows a bare "bash" would start WSL's System32/bash.exe, not Git Bash.
BASH = shutil.which("bash")


def _bash_works():
    if BASH is None:
        return False
    try:
        result = subprocess.run([BASH, "-c", "echo ok"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip() == "ok"


pytestmark = pytest.mark.skipif(not _bash_works(), reason="bash is not available")


def run_script(cwd, **env):
    full_env = {**os.environ, **env}
    for name in ("GPUS", "SMOKE", "RUN_33B", "BUDGET_MIN", "BATCH", "ACCUM"):
        if name not in env:
            full_env.pop(name, None)
    return subprocess.run([BASH, "scripts/run_server.sh"], cwd=cwd, env=full_env,
                          capture_output=True, text=True, timeout=120)


def queues(output):
    """{gpu: [runs]} from the 'GPU <n> queue: ...' lines of the plan."""
    found = {}
    for line in output.splitlines():
        if line.startswith("GPU ") and " queue: " in line:
            head, runs = line.split(" queue: ")
            found[head.split()[1]] = runs.split()
    return found


def plan(gpus, **env):
    result = run_script(ROOT, PLAN_ONLY="1", GPUS=gpus, **env)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_six_gpus_follow_the_table():
    found = queues(plan("0 1 2 3 4 5"))
    assert found["0"] == ["lora_6.7b_r16"]
    assert found["4"] == ["qlora_6.7b_r16"]
    assert found["3"][0] == "lora_1.3b_r16"
    assert found["5"] == ["zs_1.3b", "zs_6.7b", "fs3_6.7b", "cls_zs_6.7b", "cls_lora_6.7b_r16"]


@pytest.mark.parametrize("gpus", ["0", "0 1", "2 5", "0 1 2"])
def test_fewer_gpus_merge_every_run_once_in_priority_order(gpus):
    found = queues(plan(gpus))
    assert sorted(found) == sorted(gpus.split())
    every = [run for runs in found.values() for run in runs]
    assert sorted(every) == sorted(MUST + SHOULD + OPTIONAL)
    rank = {**{r: 0 for r in MUST}, **{r: 1 for r in SHOULD}, **{r: 2 for r in OPTIONAL}}
    for runs in found.values():
        assert [rank[r] for r in runs] == sorted(rank[r] for r in runs)
    assert found[gpus.split()[0]][0] == "lora_6.7b_r16"  # the main run starts first


def test_smoke_plan_uses_small_model_and_separate_dirs():
    output = plan("0", SMOKE="1", RUN_33B="1")
    commands = [line for line in output.splitlines() if "CUDA_VISIBLE_DEVICES=" in line]
    assert len(commands) == 13  # 12 queued runs + zs_33b
    for line in commands:
        assert "deepseek-coder-1.3b-instruct" in line and "6.7b-instruct" not in line
        assert "--output runs_smoke/" in line and "--limit-pairs 64" in line
    assert "--time-budget-min 2 " in output


FAKE_PYTHON = """#!/usr/bin/env bash
case "$*" in
  *preflight.py*) echo "Preflight passed."; exit 0 ;;
  *make_report.py*) echo "val table"; exit 0 ;;
esac
out=""; prev=""
for arg in "$@"; do
  if [ "$prev" = "--output" ]; then out="$arg"; fi
  prev="$arg"
done
echo "fake run $out on GPU $CUDA_VISIBLE_DEVICES"
case "$out" in *fs3_6.7b) exit 3 ;; esac
mkdir -p "$out" && echo '{}' > "$out/metrics.json"
"""


def test_queues_continue_after_a_failure_and_rerun_skips_finished(tmp_path):
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "run_server.sh")
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE_PYTHON, encoding="utf-8", newline="\n")
    fake.chmod(0o755)

    first = run_script(tmp_path, SMOKE="1", GPUS="0 1", PYTHON="./fake_python")
    assert first.returncode == 1, first.stdout + first.stderr  # fs3_6.7b (must) failed
    assert "FAILED fs3_6.7b" in first.stdout
    assert "MISSING  R3" in first.stdout
    for run in MUST + SHOULD + OPTIONAL:
        has_metrics = (tmp_path / "runs_smoke" / run / "metrics.json").exists()
        assert has_metrics == (run != "fs3_6.7b"), run
        assert (tmp_path / "logs_smoke" / f"{run}.log").exists()
    assert "on GPU 1" in (tmp_path / "logs_smoke" / "zs_6.7b.log").read_text(encoding="utf-8")
    assert not (tmp_path / "runs").exists()

    second = run_script(tmp_path, SMOKE="1", GPUS="0 1", PYTHON="./fake_python")
    assert "lora_6.7b_r16: runs_smoke/lora_6.7b_r16/metrics.json exists, skipping" in second.stdout
    assert "start fs3_6.7b" in second.stdout
