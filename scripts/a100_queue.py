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
            # first attempt (06:49 UTC): batch 16 OOM, batch 8 fits; eval batch 32 OOM, then the 3-shot
            # measurement OOMed at batch 16. Eval batch capped at 8 for the long 3-shot / longest pairs.
            cmd += ["--quantized", "--train-batches", "8", "4", "2", "--eval-batches", "8",
                    "--eval-pairs", "32", "--steps", "3", "--max-eval-batch", "8"]
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

    def wait_for_running(self):
        """A run left by an earlier queue process (restarted between runs) must finish first."""
        pattern = "src/train_repair.py|src/eval_repair.py|scripts/measure_card.py"
        while subprocess.call(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL) == 0:
            log("waiting for a run started by an earlier queue process")
            time.sleep(30)

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


def choose_prefix(left, m33, m67, load33):
    """Largest eval prefix k (pairs per split, shared by every run) and the QLoRA budget so that
    qlora_33b + its evaluation, zs_33b and fs3_33b fit in `left` minutes. The 6.7B evaluations are not
    reserved for: they run afterwards if time remains (33B cannot run anywhere else; 6.7B can).
    The first budget tried is what one epoch over 4000 pairs needs at the measured speed (<= 90 min)."""
    def evals(k):
        e33 = 2 * k / m33["eval_pairs_per_second"] / 60 * SAFETY
        f33 = 2 * k / m33["eval_pairs_per_second_3shot"] / 60 * SAFETY
        return 4 * load33 + 2 * e33 + f33 + 5
    needed = 4000 / m33["examples_per_second"] / 60 * 1.15
    for budget in (min(90, needed), 60, 45, 30):
        for k in (500, 400, 300, 250, 200, 150, 100):
            if evals(k) + budget + MARGIN_MIN <= left:
                return k, budget
    for k in (500, 300, 200, 100, 50):
        if evals(k) - 2 * load33 + MARGIN_MIN <= left:
            return k, None
    return 50, None


def train_cmd(python, size, name, m, data, budget, quantized=False, extra=(), k=500):
    batch = m["batch"]
    cmd = [python, "src/train_repair.py", "--model", SPEC[size].split("@")[0], "--revision", SPEC[size].split("@")[1],
           "--data-dir", data, "--output", f"runs/{name}", "--rank", "16", "--epochs", "1",
           "--batch-size", str(batch), "--accumulation-steps", str(max(1, 32 // batch)),
           "--eval-batch-size", str(m["eval_batch"]), "--time-budget-min", f"{budget:.0f}",
           "--eval-pairs-val", str(k), "--eval-pairs-test", str(k), "--eval-loss-pairs", str(k),
           "--seed", "42", *extra]
    return cmd + (["--quantized"] if quantized else [])


def eval_cmd(python, size, name, m, shots, quantized=False, k=500):
    batch = m.get("eval_batch_3shot", m["eval_batch"]) if shots else m["eval_batch"]
    cmd = [python, "src/eval_repair.py", "--model", SPEC[size].split("@")[0], "--revision", SPEC[size].split("@")[1],
           "--data-dir", SMALL, "--output", f"runs/{name}", "--shots", str(shots),
           "--batch-size", str(batch), "--eval-pairs-val", str(k), "--eval-pairs-test", str(k),
           "--seed", "42"]
    return cmd + (["--quantized"] if quantized else [])


def main(args):
    q = Queue(args)
    py = q.python
    log(f"queue start, deadline {q.deadline:%Y-%m-%d %H:%M} UTC, {q.minutes_left():.0f} min left")
    q.wait_for_running()
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
            load33 = m33.get("load_minutes", LOAD_MIN["33b"]) + 1  # measured load time, plus a minute
            LOAD_MIN["33b"] = load33
            k, budget33 = choose_prefix(q.minutes_left(), m33, m67, load33)
            if budget33:
                budget33 = min(budget33, args.budget_33b)
            log(f"33B plan: evaluation prefix {k} val + {k} test pairs for the remaining runs (shared with every "
                f"run: the first {k} of the small tier's 500), QLoRA budget {budget33} min")
            e33 = 2 * k / m33["eval_pairs_per_second"] / 60 * SAFETY
            f33 = 2 * k / m33["eval_pairs_per_second_3shot"] / 60 * SAFETY
            if budget33:
                q.run("qlora_33b_r16_4k", train_cmd(py, "33b", "qlora_33b_r16_4k", m33, SMALL, budget33,
                                                    quantized=True, k=k), 2 * load33 + budget33 + e33)
            else:
                log("qlora_33b_r16_4k: no time for a useful budget; not started")
            q.run("zs_33b", eval_cmd(py, "33b", "zs_33b", m33, 0, quantized=True, k=k), load33 + e33)
            q.run("fs3_33b", eval_cmd(py, "33b", "fs3_33b", m33, 3, quantized=True, k=k), load33 + f33)
            args.k67 = k
        else:
            log("33B measurement failed (see logs/measure_33b.log); 33B runs skipped")

    # d. 6.7B zero-shot and 3-shot: 500 + 500 if that still fits, else the shared prefix
    for name, shots, rate in (("zs_6.7b", 0, m67["eval_pairs_per_second"]),
                              ("fs3_6.7b", 3, m67["eval_pairs_per_second_3shot"])):
        k = 500 if LOAD_MIN["6.7b"] + 1000 / rate / 60 * SAFETY + MARGIN_MIN < q.minutes_left() else getattr(args, "k67", 500)
        q.run(name, eval_cmd(py, "6.7b", name, m67, shots, k=k), LOAD_MIN["6.7b"] + 2 * k / rate / 60 * SAFETY)

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
