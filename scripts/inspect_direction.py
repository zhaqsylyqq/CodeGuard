"""Sanity check of the fix-direction metric: print exactly which tokens s_fwd and s_bwd score.

For a few real pairs: the label span of the forward item (fixed given the buggy prompt) and of the
backward item (buggy given the fixed prompt), the number of masked prompt tokens, and whether the
span is exactly the training target (answer + end-of-turn). With --score the model is loaded and
each item is also scored token by token, one item at a time, and compared with the batched
`sequence_logprobs` used in evaluation.

    python scripts/inspect_direction.py --pairs 5
    python scripts/inspect_direction.py --pairs 5 --score --adapter runs/lora_1.3b_r16_small/adapter
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import load_pairs  # noqa: E402
from repair_utils import encode_example, encode_prompt, target_text  # noqa: E402


def check_item(tokenizer, prompt_code, target_code):
    """Label span of one scoring item and the checks on its masking."""
    item = encode_example(tokenizer, prompt_code, target_code)
    n_prompt = len(encode_prompt(tokenizer, prompt_code))
    labels = item["labels"]
    span = [t for t in labels if t != -100]
    return item, {
        "n_prompt_masked": n_prompt,
        "n_scored": len(span),
        "prompt_all_masked": labels[:n_prompt] == [-100] * n_prompt,
        "span_is_contiguous_tail": labels[n_prompt:] == item["input_ids"][n_prompt:] and -100 not in labels[n_prompt:],
        "span_is_full_target": tokenizer.decode(span) == target_text(tokenizer, prompt_code, target_code),
        "span_text": tokenizer.decode(span),
    }


def token_by_token(model, item):
    """Log-probabilities of the labelled tokens of one unpadded item, listed one by one."""
    import torch
    from torch.nn import functional as F

    from llm_utils import model_device

    ids = torch.tensor([item["input_ids"]], device=model_device(model))
    with torch.inference_mode():
        logp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), dim=-1)
    return [float(logp[pos - 1, token]) for pos, token in enumerate(item["labels"]) if token != -100]


def main(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pairs = load_pairs(args.data_dir, args.split).head(args.pairs)
    model = None
    if args.score:
        from llm_utils import load_model
        from repair_utils import sequence_logprobs

        model, tokenizer = load_model(args.model, revision=args.revision, dtype=args.dtype)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
    all_good = True
    for row in pairs.itertuples():
        print(f"\n=== pair {row.pair_id} ({row.sstub_pattern})")
        for name, prompt_code, target_code in (("s_fwd = log P(fixed | prompt(buggy))", row.buggy, row.fixed),
                                               ("s_bwd = log P(buggy | prompt(fixed))", row.fixed, row.buggy)):
            item, info = check_item(tokenizer, prompt_code, target_code)
            good = info["prompt_all_masked"] and info["span_is_contiguous_tail"] and info["span_is_full_target"]
            all_good &= good
            print(f"--- {name}: {info['n_prompt_masked']} prompt tokens masked, {info['n_scored']} tokens scored, "
                  f"prompt masked {info['prompt_all_masked']}, contiguous {info['span_is_contiguous_tail']}, "
                  f"full target {info['span_is_full_target']}")
            print(info["span_text"])
            if model is not None:
                single = token_by_token(model, item)
                batched = sequence_logprobs(model, [item, encode_example(tokenizer, target_code, prompt_code)],
                                            tokenizer.pad_token_id, batch_size=2)[0]
                print(f"    sum one by one {sum(single):.3f} | batched with padding {batched:.3f} | "
                      f"per token {sum(single) / len(single):.4f}")
    print("\nALL SPANS OK" if all_good else "\nSOME SPANS ARE WRONG")
    return 0 if all_good else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    p.add_argument("--revision", default="e063262dac8366fc1f28a4da0ff3c50ea66259ca")
    p.add_argument("--adapter")
    p.add_argument("--data-dir", default="data/processed/small")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--score", action="store_true", help="also load the model and score each item")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    return p


if __name__ == "__main__":
    sys.exit(main(parser().parse_args()))
