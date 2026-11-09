"""Evaluate full answer likelihoods for a base model or a saved LoRA adapter."""

import argparse
import json

from experiment_utils import load_examples, save_report, seed_everything
from llm_utils import MODEL_ID, load_model, predict_buggy


def run(args):
    seed_everything(args.seed)
    examples = load_examples(args.data_dir, args.split, args.limit_pairs, args.seed)
    model, tokenizer = load_model(args.model, args.quantized, args.revision)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    probabilities, truncated = predict_buggy(model, tokenizer, examples.text, args.max_length, args.batch_size)
    report = save_report(args.output, examples, probabilities, {
        "model": args.model, "revision": args.revision, "adapter": args.adapter,
        "split": args.split, "seed": args.seed, "quantized": args.quantized,
        "max_length": args.max_length, "truncated_examples": truncated,
        "score": "sum of answer token log probabilities; softmax over fixed/buggy",
        "settings": vars(args),
    })
    print(json.dumps(report, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--revision")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--output", default="runs/deepseek_zero_shot")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--adapter")
    p.add_argument("--quantized", action="store_true")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit-pairs", type=int)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
