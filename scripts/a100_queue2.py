"""Extended A100 window (until 10:00 UTC, 2026-11-14): two lanes on the one card.

Lane A: bring qlora_33b_r16_4k, zs_33b, fs3_33b from 150 to 500 test pairs (val stays 150), reusing
the 150 already done (eval_repair.extend_split); CODEGUARD_STOP_AT ends an extension early with the
pairs done. Lane B (runs at the same time; 33B evaluation uses about 25 GB, the 1.3B runs under
10 GB; only lane B trains, so there is still one training process on the card): zs_1.3b, fs3_1.3b,
lora_1.3b_r16_4k (1 epoch on the 4000 small-tier pairs).

    nohup python scripts/a100_queue2.py --lane A --deadline-utc "2026-11-14 10:00" --stop-utc "2026-11-14 09:52" &
"""

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from server_plan import MODELS  # noqa: E402

PY = sys.executable


def utc(text):
    return dt.datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)


def log(message):
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S} UTC] {message}", flush=True)


def model(size):
    name, rev = MODELS[size]
    return ["--model", name, "--revision", rev]


def lane_a():
    common = ["--data-dir", "data/processed/small", "--eval-pairs-val", "150", "--eval-pairs-test", "500",
              "--seed", "42", "--quantized"]
    return [
        ("qlora_33b_r16_4k", 15, [PY, "src/train_repair.py", *model("33b"), *common,
                                  "--output", "runs/qlora_33b_r16_4k", "--rank", "16", "--epochs", "1",
                                  "--batch-size", "8", "--accumulation-steps", "4", "--eval-batch-size", "8",
                                  "--time-budget-min", "55", "--eval-loss-pairs", "150"], True),
        ("zs_33b", 13, [PY, "src/eval_repair.py", *model("33b"), *common, "--output", "runs/zs_33b",
                        "--shots", "0", "--batch-size", "8"], True),
        ("fs3_33b", 18, [PY, "src/eval_repair.py", *model("33b"), *common, "--output", "runs/fs3_33b",
                         "--shots", "3", "--batch-size", "8"], True),
    ]


def lane_b():
    m13 = json.loads((ROOT / "logs_smoke" / "measure_1.3b.json").read_text(encoding="utf-8"))
    batch = 16 if m13["peak_train_gib"] > 10 else m13["batch"]
    common = [*model("1.3b"), "--data-dir", "data/processed/small", "--seed", "42"]
    return [
        ("zs_1.3b", 8, [PY, "src/eval_repair.py", *common, "--output", "runs/zs_1.3b", "--shots", "0",
                        "--batch-size", str(m13["eval_batch"])], False),
        ("fs3_1.3b", 10, [PY, "src/eval_repair.py", *common, "--output", "runs/fs3_1.3b", "--shots", "3",
                          "--batch-size", str(m13["eval_batch"])], False),
        ("lora_1.3b_r16_4k", 15, [PY, "src/train_repair.py", *common, "--output", "runs/lora_1.3b_r16_4k",
                                  "--rank", "16", "--epochs", "1", "--batch-size", str(batch),
                                  "--accumulation-steps", str(32 // batch), "--eval-batch-size",
                                  str(m13["eval_batch"]), "--time-budget-min", "10"], False),
    ]


def main(args):
    deadline, stop = utc(args.deadline_utc), utc(args.stop_utc)
    env = {**os.environ, "CODEGUARD_STOP_AT": str(stop.timestamp())}
    for name, estimate, cmd, extend in (lane_a() if args.lane == "A" else lane_b()):
        out = ROOT / "runs" / name
        metrics = out / "metrics.json"
        if extend:
            done = out / "metrics_test150.json"
            if metrics.exists() and not done.exists():
                shutil.move(str(metrics), str(done))  # the 150-pair metrics are kept next to the new ones
                log(f"{name}: kept the 150-test-pair metrics as metrics_test150.json")
            if metrics.exists():
                log(f"{name}: already extended, skipping")
                continue
        elif metrics.exists():
            log(f"{name}: metrics.json exists, skipping")
            continue
        left = (stop - dt.datetime.now(dt.timezone.utc)).total_seconds() / 60
        if not extend and estimate > left:
            log(f"{name}: estimated {estimate} min > {left:.0f} min before the stop time; not started")
            continue
        if extend and left < 4:
            log(f"{name}: {left:.0f} min left; not started (its 150-pair metrics stay in metrics_test150.json)")
            continue
        log(f"start {name} (lane {args.lane}): {' '.join(cmd)}")
        with open(ROOT / "logs" / f"{name}{'_test500' if extend else ''}.log", "a", encoding="utf-8") as stream:
            stream.write(f"=== {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S} UTC\n{' '.join(cmd)}\n")
            stream.flush()
            code = subprocess.call(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
        log(f"{'done' if code == 0 else 'FAILED (exit %d)' % code} {name}")
    log(f"lane {args.lane} finished; deadline {deadline:%H:%M} UTC")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lane", choices=["A", "B"], required=True)
    p.add_argument("--deadline-utc", required=True)
    p.add_argument("--stop-utc", required=True, help="no run or extension chunk may end after this")
    sys.exit(main(p.parse_args()))
