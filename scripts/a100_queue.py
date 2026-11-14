"""Unattended queue for the single-A100 server window (2026-11-14): runs sequentially, each resumable
and skipped when its metrics.json exists; nothing new starts after the deadline, and a run starts
only if its estimated duration ends before the deadline minus a margin.

Order (operator's instruction): measure 6.7B; lora_6.7b_r16_4k; measure 33B; qlora_33b_r16_4k;
zs_33b; fs3_33b; zs_6.7b; fs3_6.7b; lora_6.7b_r16 on N full-tier pairs (only if >= 45 min remain).
Every run is evaluated on the small tier's 500 val + 500 test pairs (the full tier's first 500 of
each split are the same pairs).

    nohup python scripts/a100_queue.py --deadline-utc "2026-11-14 09:30" > logs/queue.log 2>&1 &
"""

import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from server_plan import MODELS  # noqa: E402

SPEC = {size: f"{model}@{rev}" for size, (model, rev) in MODELS.items()}
SMALL, FULL = "data/processed/small", "data/processed/full"
EVAL_PAIRS = 1000            # 500 val + 500 test
MARGIN_MIN = 10              # a run must be estimated to end this long before the deadline
LOAD_MIN = {"6.7b": 2, "33b": 8}
SAFETY = 1.2


def now():
    return dt.datetime.now(dt.timezone.utc)


def log(message):
    print(f"[{now():%H:%M:%S} UTC] {message}", flush=True)


class Queue:
    def __init__(self, args):
        self.args = args
        self.deadline = dt.datetime.strptime(args.deadline_utc, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
        self.python = sys.executable

    def minutes_left(self):
        return (self.deadline - now()).total_seconds() / 60

    def measure(self, size):
        path = ROOT / "logs" / f"measure_{size}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("ok"):
                return data
        cmd = [self.python, "scripts/measure_card.py", "--model", SPEC[size], "--out", str(path)]
        if size == "33b":
            cmd += ["--quantized", "--train-batches", "16", "8", "4", "2", "--eval-batches", "8", "16", "32",
                    "--eval-pairs", "32", "--steps", "4"]
        self.execute(f"measure_{size}", cmd)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"ok": False}

    def eval_minutes(self, m, shots_also=False):
        minutes = EVAL_PAIRS / m["eval_pairs_per_second"] / 60
        if shots_also:
            minutes += EVAL_PAIRS / m["eval_pairs_per_second_3shot"] / 60
        return minutes * SAFETY

    def execute(self, name, cmd):
        log_path = ROOT / "logs" / f"{name}.log"
        log(f"start {name}: {' '.join(cmd)}")
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(f"=== {now():%Y-%m-%d %H:%M:%S} UTC\n{' '.join(cmd)}\n")
            stream.flush()
            code = subprocess.call(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        log(f"{'done' if code == 0 else 'FAILED (exit %d)' % code} {name}, log {log_path}")
        return code

    def run(self, name, cmd, estimate_min):
        if (ROOT / "runs" / name / "metrics.json").exists():
            log(f"{name}: metrics.json exists, skipping")
            return True
        left = self.minutes_left()
        if left <= 0:
            log(f"{name}: deadline passed, not started")
            return False
        if estimate_min + MARGIN_MIN > left:
            log(f"{name}: estimated {estimate_min:.0f} min + {MARGIN_MIN} margin > {left:.0f} min left; not started")
            return False
        log(f"{name}: estimated {estimate_min:.0f} min, {left:.0f} min left")
        return self.execute(name, cmd) == 0

    def wait_for_33b(self):
        marker = ROOT / "logs" / "download_33b.log"
        while self.minutes_left() > MARGIN_MIN:
            text = marker.read_text(encoding="utf-8", errors="replace") if marker.exists() else ""
            if "LARGE_DONE 0" in text:
                return True
            if "LARGE_DONE" in text:
                log("33B download FAILED; see logs/download_33b.log")
                return False
            log("waiting for the 33B download")
            time.sleep(60)
        return False


def train_cmd(python, size, name, m, data, budget, quantized=False, extra=()):
    batch = m["batch"]
    cmd = [python, "src/train_repair.py", "--model", SPEC[size].split("@")[0], "--revision", SPEC[size].split("@")[1],
           "--data-dir", data, "--output", f"runs/{name}", "--rank", "16", "--epochs", "1",
           "--batch-size", str(batch), "--accumulation-steps", str(max(1, 32 // batch)),
           "--eval-batch-size", str(m["eval_batch"]), "--time-budget-min", f"{budget:.0f}",
           "--eval-pairs-val", "500", "--eval-pairs-test", "500", "--seed", "42", *extra]
    return cmd + (["--quantized"] if quantized else [])


def eval_cmd(python, size, name, m, shots, quantized=False):
    cmd = [python, "src/eval_repair.py", "--model", SPEC[size].split("@")[0], "--revision", SPEC[size].split("@")[1],
           "--data-dir", SMALL, "--output", f"runs/{name}", "--shots", str(shots),
           "--batch-size", str(m["eval_batch"]), "--seed", "42"]
    return cmd + (["--quantized"] if quantized else [])


def main(args):
    q = Queue(args)
    py = q.python
    log(f"queue start, deadline {q.deadline:%Y-%m-%d %H:%M} UTC, {q.minutes_left():.0f} min left")
    m67 = q.measure("6.7b")
    if not m67.get("ok"):
        log("6.7B measurement failed; stopping")
        return 1
    log(f"6.7B: batch {m67['batch']}, {m67['examples_per_second']:.2f} ex/s, eval batch {m67['eval_batch']}, "
        f"{m67['eval_pairs_per_second']:.2f} pairs/s (3-shot {m67['eval_pairs_per_second_3shot']:.2f})")

    # a. 6.7B on the same 4000 pairs as the 1.3B small-tier run
    train_min = 4000 / m67["examples_per_second"] / 60 * SAFETY
    budget = max(10.0, train_min * 1.5)
    q.run("lora_6.7b_r16_4k", train_cmd(py, "6.7b", "lora_6.7b_r16_4k", m67, SMALL, budget),
          2 * LOAD_MIN["6.7b"] + train_min + q.eval_minutes(m67))

    # b, c. 33B
    if q.wait_for_33b():
        m33 = q.measure("33b")
        if m33.get("ok"):
            log(f"33B: batch {m33['batch']}, {m33['examples_per_second']:.2f} ex/s, eval batch {m33['eval_batch']}, "
                f"{m33['eval_pairs_per_second']:.2f} pairs/s (3-shot {m33['eval_pairs_per_second_3shot']:.2f})")
            eval33 = q.eval_minutes(m33)
            reserve_c = 2 * LOAD_MIN["33b"] + q.eval_minutes(m33, shots_also=True)
            budget33 = min(args.budget_33b, q.minutes_left() - MARGIN_MIN - 2 * LOAD_MIN["33b"] - eval33 - reserve_c - 5)
            if budget33 >= 20:
                q.run("qlora_33b_r16_4k", train_cmd(py, "33b", "qlora_33b_r16_4k", m33, SMALL, budget33, quantized=True),
                      2 * LOAD_MIN["33b"] + budget33 + eval33)
            else:
                log(f"qlora_33b_r16_4k: only {budget33:.0f} min of budget would remain; not started")
            q.run("zs_33b", eval_cmd(py, "33b", "zs_33b", m33, 0, quantized=True), LOAD_MIN["33b"] + eval33)
            q.run("fs3_33b", eval_cmd(py, "33b", "fs3_33b", m33, 3, quantized=True),
                  LOAD_MIN["33b"] + EVAL_PAIRS / m33["eval_pairs_per_second_3shot"] / 60 * SAFETY)
        else:
            log("33B measurement failed (see logs/measure_33b.log); 33B runs skipped")

    # d. 6.7B zero-shot and 3-shot
    q.run("zs_6.7b", eval_cmd(py, "6.7b", "zs_6.7b", m67, 0), LOAD_MIN["6.7b"] + q.eval_minutes(m67))
    q.run("fs3_6.7b", eval_cmd(py, "6.7b", "fs3_6.7b", m67, 3),
          LOAD_MIN["6.7b"] + EVAL_PAIRS / m67["eval_pairs_per_second_3shot"] / 60 * SAFETY)

    # e. 6.7B on N full-tier pairs, sized from the measured speed and the time left
    left = q.minutes_left()
    if left >= 45:
        budget = left - MARGIN_MIN - 2 * LOAD_MIN["6.7b"] - q.eval_minutes(m67) - 5
        n = min(50000, math.floor(0.85 * m67["examples_per_second"] * budget * 60 / 1000) * 1000)
        if n >= 1000:
            log(f"lora_6.7b_r16: N = {n} pairs (budget {budget:.0f} min)")
            q.run("lora_6.7b_r16", train_cmd(py, "6.7b", "lora_6.7b_r16", m67, FULL, budget,
                                             extra=("--train-pairs", str(n))),
                  2 * LOAD_MIN["6.7b"] + budget + q.eval_minutes(m67))
    else:
        log(f"lora_6.7b_r16: only {left:.0f} min left (< 45); not started")
    log("queue finished")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deadline-utc", required=True, help='e.g. "2026-11-14 09:30"')
    p.add_argument("--budget-33b", type=float, default=90.0)
    return p


if __name__ == "__main__":
    sys.exit(main(parser().parse_args()))
