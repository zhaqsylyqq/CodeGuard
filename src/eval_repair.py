"""Evaluate bug repair zero-shot, few-shot, or with a LoRA adapter; writes predictions and metrics.json.

Every run writes predictions for every requested split (val and test). Choices are made on val only;
see section 6.5 of docs/AGENT_GUIDE.md and src/make_report.py.
"""

import argparse
import json
import time
from pathlib import Path

from experiment_utils import DEFAULT_DATA_DIR, load_pairs, run_metadata, seed_everything, write_json
from repair_utils import choose_shots, evaluate_split, summarize

SUBSET_SEED = 42  # --limit-pairs subsets are the same pairs for every run, so runs can be paired


def evaluate_run(model, tokenizer, data_dir, splits, output, shots=(), batch_size=8, limit_pairs=None,
                 direction=True):
    """Write {split}_predictions.parquet per split and return {split: summary}.

    A split whose prediction file already exists is only re-summarized, so a crashed evaluation
    continues where it stopped.
    """
    import pandas as pd

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in splits:
        path = output / f"{split}_predictions.parquet"
        if path.exists():
            table = pd.read_parquet(path)
            results[split] = {**summarize(table), "eval_seconds": None, "resumed": True}
            continue
        pairs = load_pairs(data_dir, split, limit_pairs, SUBSET_SEED)
        start = time.monotonic()
        table, summary = evaluate_split(model, tokenizer, pairs, shots, batch_size, direction)
        seconds = time.monotonic() - start
        table.to_parquet(path, index=False)
        results[split] = {**summary, "eval_seconds": seconds, "pairs_per_second": len(pairs) / seconds}
        print(f"{split}: exact match {summary['overall']['exact_match']:.4f} "
              f"on {len(pairs)} pairs in {seconds / 60:.1f} min", flush=True)
    return results


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def run(args):
    output = Path(args.output)
    if (output / "metrics.json").exists():
        print(f"{output / 'metrics.json'} exists; nothing to do.")
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    seed_everything(args.seed)
    from llm_utils import load_model

    shots, shot_ids = (), []
    if args.shots:
        chosen = choose_shots(load_pairs(args.data_dir, "train"), k=args.shots, seed=args.seed)
        shots = list(zip(chosen["buggy"], chosen["fixed"]))
        shot_ids = [int(i) for i in chosen["pair_id"]]
    model, tokenizer = load_model(args.model, args.quantized, args.revision, args.dtype, args.multi_gpu)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.config.use_cache = True
    start = time.monotonic()
    results = evaluate_run(model, tokenizer, args.data_dir, args.splits, output, shots, args.batch_size,
                           args.limit_pairs, not args.no_direction)
    method = "lora" if args.adapter else ("zero-shot" if not args.shots else f"{args.shots}-shot")
    report = {
        "run": output.name, "task": "repair", "method": method, "model": args.model,
        "revision": args.revision, "adapter": args.adapter, "quantized": args.quantized,
        "shots": args.shots, "shot_pair_ids": shot_ids, "splits": results,
        "eval_seconds": time.monotonic() - start, **count_parameters(model), **run_metadata(args),
    }
    write_json(output / "metrics.json", report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    p.add_argument("--revision")
    p.add_argument("--adapter", help="directory of a saved LoRA adapter")
    p.add_argument("--shots", type=int, choices=[0, 3], default=0)
    p.add_argument("--quantized", action="store_true", help="load the base model in 4-bit NF4")
    p.add_argument("--multi-gpu", action="store_true", help='spread the model over all visible GPUs (device_map="auto")')
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit-pairs", type=int, help="evaluate a fixed random subset of this many pairs per split")
    p.add_argument("--no-direction", action="store_true", help="skip the fix-direction metric")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
