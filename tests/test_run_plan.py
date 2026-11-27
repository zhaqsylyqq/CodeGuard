"""Server plan (scripts/server_plan.py) and the queue runner scripts/run_server.sh with a fake python.

Nothing here loads a model. The bash tests are skipped where bash is not available.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import server_plan  # noqa: E402

SCRIPT = ROOT / "scripts" / "run_server.sh"
ASSUMED = ["--assume-speed", "3", "--assume-eval-speed", "1", "--assume-eval-speed-3shot", "0.6",
           "--assume-33b-speed", "0.5", "--assume-33b-eval-speed", "0.25", "--assume-33b-eval-speed-3shot", "0.15"]


def build(*extra, timing="none"):
    args = server_plan.parser().parse_args(["--timing-json", str(timing), *extra])
    plan, speeds, large = server_plan.build(args)
    return plan, speeds, large, args


def queues(plan):
    return {"+".join(q["devices"]): [s["name"] for s in q["runs"]] for q in plan["queues"]}


def every_run(plan):
    return [s for q in plan["queues"] for s in q["runs"]]


def test_six_gpus_follow_the_architects_layout_at_540_min():
    plan, speeds, _, _ = build("--window-min", "540", *ASSUMED)
    found = queues(plan)
    assert found["0"] == ["zs_6.7b", "fs3_6.7b", "lora_6.7b_r16_4k"]
    assert found["1"] == ["lora_6.7b_r16"]
    assert found["2+3"] == ["zs_33b", "fs3_33b"]
    assert found["4+5"] == ["qlora_33b_r16"]
    assert plan["n_main"] == 50000  # 0.85 x 3/s x budget is above the 50k cap


@pytest.mark.parametrize("window", [210, 540])
def test_every_queue_fits_the_window_with_margin_and_no_1_3b_runs(window):
    plan, speeds, _, args = build("--window-min", str(window), *ASSUMED)
    for queue in plan["queues"]:
        assert server_plan.used_minutes(queue, speeds) <= window - 20 + 1e-9
    for spec in every_run(plan):
        assert spec["size"] in ("6.7b", "33b")
        assert "1.3b" not in " ".join(server_plan.command(spec, args, speeds))


def test_main_run_size_and_eval_prefixes():
    plan, speeds, _, args = build("--window-min", "210", *ASSUMED)
    main = next(s for s in every_run(plan) if s["name"] == "lora_6.7b_r16")
    expected = min(50000, int(0.85 * 3 * plan["main_budget"] * 60) // 1000 * 1000)
    assert main["train_pairs"] == plan["n_main"] == expected and plan["n_main"] % 1000 == 0
    cmd = server_plan.command(main, args, speeds)
    assert cmd[cmd.index("--epochs") + 1] == "1" and cmd[cmd.index("--train-pairs") + 1] == str(expected)
    assert cmd[cmd.index("--eval-pairs-val") + 1] == "500" and cmd[cmd.index("--eval-pairs-test") + 1] == "2000"
    zs33 = next(s for s in every_run(plan) if s["name"] == "zs_33b")
    cmd = server_plan.command(zs33, args, speeds)
    assert "--multi-gpu" in cmd and "--quantized" in cmd and cmd[cmd.index("--eval-pairs-test") + 1] == "1000"


def test_33b_training_requests_at_least_4000_pairs():
    plan, speeds, _, args = build("--window-min", "210", *ASSUMED)
    t33 = next(s for s in every_run(plan) if s["name"] == "qlora_33b_r16")
    assert t33["train_pairs"] >= 4000 and "train_examples_seen" in t33["note"]
    cmd = server_plan.command(t33, args, speeds)
    assert "--multi-gpu" in cmd and "--quantized" in cmd


def test_dropped_runs_are_reported_not_lost():
    plan, _, _, _ = build("--window-min", "210", *ASSUMED)
    planned = {s["name"] for s in every_run(plan)}
    dropped = {name for name, _ in plan["dropped"]}
    assert planned | dropped >= {"zs_6.7b", "fs3_6.7b", "lora_6.7b_r16", "lora_6.7b_r16_4k", "zs_33b", "fs3_33b",
                                 "qlora_33b_r16", "lora_6.7b_r8", "lora_6.7b_r32"}
    assert not planned & dropped


@pytest.mark.parametrize("gpus, first_groups", [("0 1 2 3 4", {"0+1": "zs_33b", "2": "lora_6.7b_r16",
                                                                "3+4": "qlora_33b_r16"}),
                                                 ("0 1 2", {"0+1": "zs_33b", "2": "lora_6.7b_r16"}),
                                                 ("7", {"7": "lora_6.7b_r16"})])
def test_fewer_gpus_follow_the_fallback_priority(gpus, first_groups):
    plan, _, _, _ = build("--window-min", "540", "--gpus", gpus, *ASSUMED)
    found = queues(plan)
    for devices, first in first_groups.items():
        assert found[devices][0] == first
    if len(gpus.split()) < 2:
        assert all(not s["pair"] for s in every_run(plan))


def test_with_33b_0_gives_the_four_cards_to_6_7b_runs():
    plan, _, large, _ = build("--window-min", "210", "--with-33b", "0", *ASSUMED[:6])
    found = queues(plan)
    assert not large
    assert [found[g] for g in ("2", "3", "4", "5")] == [["lora_6.7b_r8"], ["lora_6.7b_r32"], ["qlora_6.7b_r16"],
                                                        ["lora_6.7b_r16_s43"]]


def test_failed_33b_preflight_switches_to_the_fallback(tmp_path):
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps({"main": {"examples_per_second": 3, "eval_pairs_per_second": 1,
                                           "eval_pairs_per_second_3shot": 0.6},
                                  "33b": {"ok": False, "fits_window": False}}), encoding="utf-8")
    plan, speeds, large, _ = build("--window-min", "210", timing=timing)
    assert not large and "measured" in speeds.source
    assert any("WITH_33B=0" in note for note in plan["notes"])
    assert "qlora_6.7b_r16" in {s["name"] for s in every_run(plan)}


def test_unknown_speeds_stop_the_plan():
    with pytest.raises(SystemExit, match="preflight"):
        build("--window-min", "210")


def test_smoke_exercises_the_two_gpu_path_only_with_two_gpus():
    plan, _, _, args = build("--smoke", "1", "--gpus", "0 1", "--runs", "runs_smoke")
    late = [q for q in plan["queues"] if q.get("after_others")]
    assert len(late) == 1 and late[0]["devices"] == ["0", "1"]
    assert [s["name"] for s in late[0]["runs"]] == ["zs_33b", "fs3_33b", "qlora_33b_r16"]
    for spec in every_run(plan):
        cmd = " ".join(server_plan.command(spec, args, None))
        assert "1.3b-instruct" in cmd and "--output runs_smoke/" in cmd
    one, _, _, _ = build("--smoke", "1", "--gpus", "0")
    assert not any(s["pair"] for s in every_run(one))
    assert any("SKIPPED" in note for note in one["notes"])


# ---------------------------------------------------------------- bash


BASH = shutil.which("bash")  # full path: on Windows a bare "bash" may start WSL instead of Git Bash


def _bash_works():
    if BASH is None:
        return False
    try:
        result = subprocess.run([BASH, "-c", "echo ok"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip() == "ok"


bash_only = pytest.mark.skipif(not _bash_works(), reason="bash is not available")


def run_script(cwd, **env):
    full_env = {**os.environ, "PLAN_PYTHON": sys.executable, **env}
    for name in ("GPUS", "SMOKE", "WITH_33B", "RUN_33B", "WINDOW_MIN", "BATCH", "ACCUM", "PYTHON"):
        if name not in env:
            full_env.pop(name, None)
    return subprocess.run([BASH, "scripts/run_server.sh"], cwd=cwd, env=full_env, capture_output=True, text=True,
                          timeout=300)


@bash_only
def test_plan_only_prints_the_plan_with_assumed_speeds():
    result = run_script(ROOT, PLAN_ONLY="1", WINDOW_MIN="210", PYTHON=sys.executable, ASSUME_SPEED="3",
                        ASSUME_EVAL_SPEED="1", ASSUME_EVAL_SPEED_3SHOT="0.6", ASSUME_33B_SPEED="0.5",
                        ASSUME_33B_EVAL_SPEED="0.25", ASSUME_33B_EVAL_SPEED_3SHOT="0.15")
    assert result.returncode == 0, result.stderr
    assert "ASSUMED speeds, NOT measured" in result.stdout and "Commands:" in result.stdout
    assert "GPU 1: lora_6.7b_r16" in result.stdout


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


@bash_only
def test_queues_continue_after_a_failure_and_rerun_skips_finished(tmp_path):
    (tmp_path / "scripts").mkdir()
    for name in ("run_server.sh", "server_plan.py"):
        shutil.copy(ROOT / "scripts" / name, tmp_path / "scripts" / name)
    fake = tmp_path / "fake_python"
    fake.write_text(FAKE_PYTHON, encoding="utf-8", newline="\n")
    fake.chmod(0o755)

    first = run_script(tmp_path, SMOKE="1", GPUS="0 1", PYTHON="./fake_python")
    assert first.returncode == 1, first.stdout + first.stderr  # fs3_6.7b (planned) failed
    assert "FAILED fs3_6.7b" in first.stdout and "MISSING  fs3_6.7b" in first.stdout
    planned = ["zs_6.7b", "fs3_6.7b", "lora_6.7b_r16", "lora_6.7b_r16_4k", "lora_6.7b_r8", "lora_6.7b_r32",
               "zs_33b", "fs3_33b", "qlora_33b_r16"]
    for run in planned:
        assert (tmp_path / "runs_smoke" / run / "metrics.json").exists() == (run != "fs3_6.7b"), run
        assert (tmp_path / "logs_smoke" / f"{run}.log").exists()
    assert "on GPU 0,1" in (tmp_path / "logs_smoke" / "qlora_33b_r16.log").read_text(encoding="utf-8")
    assert not (tmp_path / "runs").exists()

    second = run_script(tmp_path, SMOKE="1", GPUS="0 1", PYTHON="./fake_python")
    assert "lora_6.7b_r16: runs_smoke/lora_6.7b_r16/metrics.json exists, skipping" in second.stdout
    assert "start fs3_6.7b" in second.stdout
