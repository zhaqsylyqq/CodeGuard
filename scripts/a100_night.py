"""Unattended overnight queue on the single A100 (2026-11-16 -> 17, deadline 04:00 UTC = 09:00 local).

One job at a time. Every run is skipped when its metrics.json is complete, a failed run is logged and
the queue moves on, and a run starts only if its estimated duration (with evaluation) ends before the
deadline minus a margin. Prediction files that are a prefix of the requested pairs are extended
(eval_repair.extend_split); the metrics they had are kept as metrics_test<n>.json. At the end the
queue packs runs (without checkpoints), logs and results into /workspace/codeguard_night_results.tar.gz
and writes logs/NIGHT_QUEUE_DONE.

Blocks (operator's order): 1 zs_33b / fs3_33b test -> 500; 2 lora_6.7b_r16_4k seeds 43, 44;
3 lora_6.7b_r8_4k, lora_6.7b_r32_4k, qlora_6.7b_r16_4k; 4 cls_zs_6.7b, cls_lora_6.7b_r16;
5 lora_6.7b_r16_50k; 6 zs_6.7b, fs3_6.7b, lora_6.7b_r16_50k on the first 2000 full-tier test pairs;
7 qlora_33b_r16_<N>k on a full-tier prefix sized from the measured 33B speed and the time left.

    nohup python scripts/a100_night.py --deadline-utc "2026-11-17 04:00" >> logs/night_queue.log 2>&1 &
"""

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from server_plan import MODELS  # noqa: E402

PY = sys.executable
SMALL, FULL = "data/processed/small", "data/processed/full"
PACK_MARGIN_MIN = 15        # nothing may end later than this before the deadline (packing time)
MARGIN_33B_MIN = 30         # block 7 must finish this long before the deadline (operator's rule)
SPEED_33B = 1.66            # examples/s of qlora_33b_r16_4k over its whole epoch (2026-11-14)
EVAL_33B_PAIRS_PER_S = 0.45  # qlora_33b_r16_4k evaluation, alone on the card (2026-11-14)


def now():
    return dt.datetime.now(dt.timezone.utc)


def log(message):
    print(f"[{now():%Y-%m-%d %H:%M:%S} UTC] {message}", flush=True)


def model(size):
    name, rev = MODELS[size]
    return ["--model", name, "--revision", rev]


def test_rows(name):
    path = ROOT / "runs" / name / "test_predictions.parquet"
    if not path.exists():
        return 0
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


class Night:
    def __init__(self, deadline):
        self.deadline = deadline
        self.stop_at = deadline - dt.timedelta(minutes=PACK_MARGIN_MIN)
        self.env = {**os.environ, "CODEGUARD_STOP_AT": str(self.stop_at.timestamp()),
                    "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}

    def left(self):
        return (self.stop_at - now()).total_seconds() / 60

    def run(self, name, cmd, estimate, test_target=None):
        """Run one job. test_target: number of test pairs metrics.json must cover (extension runs)."""
        out = ROOT / "runs" / name
        metrics = out / "metrics.json"
        if metrics.exists():
            have = test_rows(name)
            if test_target is None or have >= test_target:
                log(f"{name}: metrics.json complete, skipping")
                return True
            kept = out / f"metrics_test{have}.json"
            shutil.move(str(metrics), str(kept))
            log(f"{name}: {have} test pairs < {test_target}; kept its metrics as {kept.name}, extending")
        if estimate > self.left():
            log(f"{name}: estimated {estimate:.0f} min > {self.left():.0f} min left; not started")
            self.restore(out)
            return False
        log(f"start {name} (estimated {estimate:.0f} min, {self.left():.0f} min left): {' '.join(cmd)}")
        with open(ROOT / "logs" / f"{name}.log", "a", encoding="utf-8") as stream:
            stream.write(f"=== {now():%Y-%m-%d %H:%M:%S} UTC (night queue)\n{' '.join(cmd)}\n")
            stream.flush()
            code = subprocess.call(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=self.env)
        if code == 0 and metrics.exists():
            log(f"done {name} (test pairs: {test_rows(name)})")
            return True
        log(f"FAILED {name} (exit {code}); see logs/{name}.log; queue continues")
        self.restore(out)
        return False

    @staticmethod
    def restore(out):
        """If a run did not write new metrics, put the newest kept ones back so metrics.json stays valid."""
        if (out / "metrics.json").exists():
            return
        kept = sorted(out.glob("metrics_test*.json"), key=lambda p: p.stat().st_mtime)
        if kept:
            shutil.copy(str(kept[-1]), str(out / "metrics.json"))
            log(f"{out.name}: restored {kept[-1].name} as metrics.json (predictions may hold more pairs)")


def train67(name, data, rank=16, seed=42, quantized=False, budget=25, extra=()):
    cmd = [PY, "src/train_repair.py", *model("6.7b"), "--data-dir", data, "--output", f"runs/{name}",
           "--rank", str(rank), "--epochs", "1", "--batch-size", "32", "--accumulation-steps", "1",
           "--eval-batch-size", "16", "--time-budget-min", str(budget), "--seed", str(seed), *extra]
    return cmd + (["--quantized"] if quantized else [])


def eval_cmd(size, name, shots, data, val, test, batch, quantized=False):
    cmd = [PY, "src/eval_repair.py", *model(size), "--data-dir", data, "--output", f"runs/{name}",
           "--shots", str(shots), "--batch-size", str(batch), "--eval-pairs-val", str(val),
           "--eval-pairs-test", str(test), "--seed", "42"]
    return cmd + (["--quantized"] if quantized else [])


def plan_33b(left_min):
    """N (pairs, thousands) and training budget for block 7 so that it ends MARGIN_33B_MIN before the deadline."""
    evaluation = (150 + 500) / EVAL_33B_PAIRS_PER_S / 60 * 1.2
    fixed = 2 + evaluation + 6 + 5  # two loads, evaluation, four eval-loss passes, saving
    budget = left_min + PACK_MARGIN_MIN - MARGIN_33B_MIN - fixed
    n = min(50000, math.floor(SPEED_33B * 0.9 * budget * 60 / 1000) * 1000)
    return n, budget, fixed


def main(args):
    deadline = dt.datetime.strptime(args.deadline_utc, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
    q = Night(deadline)
    log(f"night queue start; deadline {deadline:%Y-%m-%d %H:%M} UTC, runs must end by {q.stop_at:%H:%M} UTC")

    # 1. 33B test pairs 150/342 -> 500 (val stays 150)
    q.run("zs_33b", eval_cmd("33b", "zs_33b", 0, SMALL, 150, 500, 8, True), 12, test_target=500)
    q.run("fs3_33b", eval_cmd("33b", "fs3_33b", 3, SMALL, 150, 500, 8, True), 25, test_target=500)
    # 2. seed noise of the 6.7B 4k run (same settings as seed 42)
    for seed in (43, 44):
        q.run(f"lora_6.7b_r16_4k_s{seed}", train67(f"lora_6.7b_r16_4k_s{seed}", SMALL, seed=seed, budget=18), 22)
    # 3. rank and quantization on the same 4000 pairs
    q.run("lora_6.7b_r8_4k", train67("lora_6.7b_r8_4k", SMALL, rank=8), 22)
    q.run("lora_6.7b_r32_4k", train67("lora_6.7b_r32_4k", SMALL, rank=32), 24)
    q.run("qlora_6.7b_r16_4k", train67("qlora_6.7b_r16_4k", SMALL, quantized=True, budget=35), 40)
    # 4. isolated detection (legacy scripts; scored on val)
    q.run("cls_zs_6.7b", [PY, "src/llm_baseline.py", *model("6.7b"), "--data-dir", SMALL, "--split", "val",
                          "--output", "runs/cls_zs_6.7b", "--batch-size", "16"], 10)
    q.run("cls_lora_6.7b_r16", [PY, "src/train_lora.py", *model("6.7b"), "--data-dir", SMALL,
                                "--output", "runs/cls_lora_6.7b_r16", "--rank", "16", "--epochs", "1",
                                "--batch-size", "16", "--accumulation-steps", "2", "--time-budget-min", "45"], 55)
    # 5. all 50,000 full-tier pairs; evaluated on the first 2000 full-tier test pairs right away (block 6)
    q.run("lora_6.7b_r16_50k", train67("lora_6.7b_r16_50k", FULL, budget=150,
                                       extra=("--eval-pairs-val", "500", "--eval-pairs-test", "2000")), 175,
          test_target=2000)
    # 6. headline 6.7B comparison on the first 2000 full-tier test pairs (small test = their first 500)
    q.run("zs_6.7b", eval_cmd("6.7b", "zs_6.7b", 0, FULL, 500, 2000, 16), 15, test_target=2000)
    q.run("fs3_6.7b", eval_cmd("6.7b", "fs3_6.7b", 3, FULL, 500, 2000, 16), 25, test_target=2000)
    # 7. 33B QLoRA on a full-tier prefix sized from the measured speed and the time left
    n, budget, fixed = plan_33b(q.left())
    if n >= 8000:
        name = f"qlora_33b_r16_{n // 1000}k"
        log(f"block 7: N = {n} pairs (training budget {budget:.0f} min + {fixed:.0f} min load/eval/saving, "
            f"speed {SPEED_33B} ex/s x 0.9)")
        cmd = [PY, "src/train_repair.py", *model("33b"), "--data-dir", FULL, "--output", f"runs/{name}",
               "--rank", "16", "--epochs", "1", "--batch-size", "8", "--accumulation-steps", "4",
               "--eval-batch-size", "8", "--time-budget-min", f"{budget:.0f}", "--train-pairs", str(n),
               "--eval-pairs-val", "150", "--eval-pairs-test", "500", "--eval-loss-pairs", "150",
               "--quantized", "--seed", "42"]
        q.run(name, cmd, budget + fixed)
    else:
        log(f"block 7: only N = {n} pairs would fit (< 8000); not started")
    pack()
    return 0


def pack():
    out = ROOT / "codeguard_night_results.tar.gz"
    subprocess.call("nvidia-smi > logs/nvidia_smi_night_end.txt 2>&1", shell=True, cwd=ROOT)
    code = subprocess.call(["tar", "--exclude=checkpoints", "--exclude=__pycache__", "-czf", str(out),
                            "runs", "logs", "results"], cwd=ROOT)
    log(f"packed {out} (tar exit {code}, {out.stat().st_size / 2**20:.0f} MiB)" if out.exists() else "pack FAILED")
    (ROOT / "logs" / "NIGHT_QUEUE_DONE").write_text(f"{now():%Y-%m-%d %H:%M:%S} UTC\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deadline-utc", required=True)
    sys.exit(main(p.parse_args()))
