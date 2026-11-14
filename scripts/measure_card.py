"""Quick measurement on the GPU before the real runs: the largest train batch that fits the longest
examples, training examples/s, and the fastest evaluation batch (prefilled generation + direction).

    python scripts/measure_card.py --model ID@SHA --out logs/measure_6.7b.json
    python scripts/measure_card.py --model ID@SHA --quantized --train-batches 16 8 4 2 --eval-batches 8 16 32

Uses the same functions as preflight.py; writes one JSON file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import preflight  # noqa: E402
from download_models import split_spec  # noqa: E402


def main(args):
    import torch

    from experiment_utils import load_pairs
    from train_repair import build_lora_model

    start = time.monotonic()
    model_id, revision = split_spec(args.model)
    model, tokenizer = build_lora_model(model_id, revision, rank=16, quantized=args.quantized)
    result = {"model": args.model, "quantized": args.quantized, "load_minutes": (time.monotonic() - start) / 60,
              "gpu": torch.cuda.get_device_name(0)}
    measured = preflight.measure_training(model, tokenizer, load_pairs(args.data_dir, "train"), args.train_batches,
                                          args.steps, args.max_length, label="train timing")
    if measured is None:
        result["ok"] = False
    else:
        result["batch"], result["examples_per_second"] = measured
        result["peak_train_gib"] = torch.cuda.max_memory_allocated() / 2**30
        speeds = {}
        for batch in args.eval_batches:
            try:
                speeds[batch] = preflight.measure_eval(model, tokenizer, args.data_dir, args.eval_pairs, batch)
            except torch.cuda.OutOfMemoryError:
                speeds[batch] = None
                preflight.free_cuda()
                print(f"      eval batch {batch}: CUDA out of memory", flush=True)
        good = {b: s for b, s in speeds.items() if s}
        result["eval_pairs_per_second_by_batch"] = speeds
        result["eval_batch"] = max(good, key=good.get)
        result["eval_pairs_per_second"] = good[result["eval_batch"]]
        result["eval_pairs_per_second_3shot"] = preflight.measure_eval(
            model, tokenizer, args.data_dir, args.eval_pairs, result["eval_batch"], 3)
        result["ok"] = True
    result["minutes_total"] = (time.monotonic() - start) / 60
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["ok"] else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--quantized", action="store_true")
    p.add_argument("--data-dir", default="data/processed/small")
    p.add_argument("--train-batches", type=int, nargs="+", default=[32, 16, 8])
    p.add_argument("--eval-batches", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--eval-pairs", type=int, default=64)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    sys.exit(main(parser().parse_args()))
