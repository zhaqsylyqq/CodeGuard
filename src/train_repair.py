"""LoRA / QLoRA fine-tuning of a code LLM for single-statement bug repair, then evaluation on val and test.

Safe to re-run: exits at once if metrics.json exists, resumes from the last checkpoint, and if the
adapter is already saved it only runs the evaluation.
"""

import argparse
import json
import math
import time
from pathlib import Path

from eval_repair import SUBSET_SEED, add_subset_flags, count_parameters, eval_prefixes, evaluate_run
from experiment_utils import DEFAULT_DATA_DIR, load_pairs, run_metadata, seed_everything, write_json
from repair_utils import GENERATION_RULE, encode_example


def lora_target_modules(model):
    """Names of all linear projections inside the decoder blocks (attention and MLP), not the LM head."""
    import torch

    head = model.get_output_embeddings()
    names = {name.split(".")[-1] for name, module in model.named_modules()
             if isinstance(module, torch.nn.Linear) and module is not head}
    if not names:
        raise ValueError("No linear layers found for LoRA")
    return sorted(names)


def build_lora_model(model_id, revision=None, rank=16, quantized=False, dtype="auto"):
    """Base model (bf16, or 4-bit NF4 for QLoRA) with fresh LoRA adapters; gradient checkpointing on."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from llm_utils import load_model

    model, tokenizer = load_model(model_id, quantized, revision, dtype)
    model.config.use_cache = False
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    config = LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                        target_modules=lora_target_modules(model), bias="none")
    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, tokenizer


def encode_pairs(tokenizer, pairs, max_length):
    """Training items; examples longer than max_length are dropped (never truncated) and counted."""
    items, dropped = [], 0
    for buggy, fixed in zip(pairs["buggy"], pairs["fixed"]):
        item = encode_example(tokenizer, buggy, fixed)
        if len(item["input_ids"]) > max_length:
            dropped += 1
        else:
            items.append(item)
    return items, dropped


def last_checkpoint(directory):
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(directory)) if Path(directory).is_dir() else None


def train(args, output):
    """Train and save the adapter; returns training facts for metrics.json."""
    import torch
    from transformers import Trainer, TrainingArguments

    from llm_utils import AnswerCollator, TimeBudgetCallback, resolve_dtype

    train_pairs = load_pairs(args.data_dir, "train", args.limit_pairs, SUBSET_SEED, first=args.train_pairs)
    if args.limit_pairs is None:  # eval loss on a prefix of val, so it is a subset of the evaluated pairs
        eval_pairs = load_pairs(args.data_dir, "val", first=args.eval_loss_pairs)
    else:
        eval_pairs = load_pairs(args.data_dir, "val", min(args.eval_loss_pairs, args.limit_pairs), SUBSET_SEED)
    model, tokenizer = build_lora_model(args.model, args.revision, args.rank, args.quantized, args.dtype)
    train_items, dropped_train = encode_pairs(tokenizer, train_pairs, args.max_length)
    eval_items, dropped_eval = encode_pairs(tokenizer, eval_pairs, args.max_length)
    print(f"train examples {len(train_items)} (dropped for length {dropped_train}), "
          f"eval-loss examples {len(eval_items)} (dropped {dropped_eval})", flush=True)

    steps_per_epoch = math.ceil(len(train_items) / (args.batch_size * args.accumulation_steps))
    eval_every = max(1, steps_per_epoch // 4)  # four evaluations per epoch
    dtype = resolve_dtype(args.dtype)
    on_gpu = torch.cuda.is_available()
    settings = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0,  # warmup_steps < 1 = ratio
        eval_strategy="steps", eval_steps=eval_every, save_strategy="steps", save_steps=eval_every,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, bf16=on_gpu and dtype == torch.bfloat16,
        fp16=on_gpu and dtype == torch.float16, train_sampling_strategy="group_by_length",
        logging_steps=10, report_to="none", seed=args.seed, data_seed=args.seed,
        dataloader_pin_memory=on_gpu, remove_unused_columns=False, label_names=["labels"],
    )
    budget = TimeBudgetCallback(args.time_budget_min, output / "train_time.json")
    trainer = Trainer(model=model, args=settings, train_dataset=train_items, eval_dataset=eval_items,
                      data_collator=AnswerCollator(tokenizer.pad_token_id), processing_class=tokenizer,
                      callbacks=[budget])
    resume = last_checkpoint(output / "checkpoints")
    if resume:
        print(f"resuming from {resume}", flush=True)
    else:
        # No checkpoint (first attempt, or the checkpoints were lost with a Colab session): training
        # starts over, so its time budget starts over too.
        (output / "train_time.json").unlink(missing_ok=True)
    start = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume)
    train_seconds = time.monotonic() - start

    adapter = output / "adapter"
    trainer.model.save_pretrained(adapter)  # the best checkpoint by eval_loss is loaded at the end
    tokenizer.save_pretrained(adapter)
    info = {
        "train_pairs": len(train_pairs), "train_examples": len(train_items),
        "dropped_for_length": {"train": dropped_train, "eval_loss_subset": dropped_eval},
        "eval_loss_pairs": len(eval_items), "steps_per_epoch": steps_per_epoch, "eval_every_steps": eval_every,
        "global_step": trainer.state.global_step, "epochs_done": trainer.state.epoch,
        # examples the optimizer actually saw (less than train_examples when the time budget stopped it)
        "train_examples_seen": int(min(trainer.state.global_step * args.batch_size * args.accumulation_steps,
                                       round((trainer.state.epoch or 0) * len(train_items)))),
        "stopped_by_time_budget": budget.stopped_early, "resumed_from": resume,
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
        "train_seconds": train_seconds, "train_seconds_all_attempts": budget.elapsed(),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "log_history": trainer.state.log_history, **count_parameters(model),
        "adapter_bytes": sum(f.stat().st_size for f in adapter.iterdir() if f.is_file()),
    }
    write_json(output / "train_info.json", info)
    return info


def free_gpu_memory():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_trained(args, output):
    """Base model plus the saved adapter, for evaluation after a crash between training and eval."""
    from peft import PeftModel

    from llm_utils import load_model

    model, tokenizer = load_model(args.model, args.quantized, args.revision, args.dtype)
    return PeftModel.from_pretrained(model, output / "adapter"), tokenizer


def run(args):
    from llm_utils import require_single_gpu

    output = Path(args.output)
    if (output / "metrics.json").exists():
        print(f"{output / 'metrics.json'} exists; nothing to do.")
        return json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    require_single_gpu()
    seed_everything(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "train_info.json").exists() and (output / "adapter").is_dir():
        print("adapter already trained; evaluating only", flush=True)
    else:
        train(args, output)
        free_gpu_memory()
    info = json.loads((output / "train_info.json").read_text(encoding="utf-8"))
    # A fresh load of base + saved adapter: exactly what eval_repair.py --adapter would evaluate.
    model, tokenizer = load_trained(args, output)
    model.config.use_cache = True
    start = time.monotonic()
    results = evaluate_run(model, tokenizer, args.data_dir, args.splits, output, (), args.eval_batch_size,
                           args.limit_pairs, not args.no_direction, eval_prefixes(args))
    report = {
        "run": output.name, "task": "repair", "method": "qlora" if args.quantized else "lora",
        "model": args.model, "revision": args.revision, "adapter": str(output / "adapter"),
        "rank": args.rank, "quantized": args.quantized, "shots": 0, "generation": GENERATION_RULE,
        "splits": results,
        "eval_seconds": time.monotonic() - start, **info, **run_metadata(args),
    }
    write_json(output / "metrics.json", report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct")
    p.add_argument("--revision")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--output", required=True)
    p.add_argument("--rank", type=int, choices=[8, 16, 32], default=16)
    p.add_argument("--quantized", action="store_true", help="QLoRA: 4-bit NF4 base model")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=8, help="per-device batch; batch x accumulation = 32")
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8, help="batch size for generation after training")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--limit-pairs", type=int, help="use a fixed random subset of this many pairs per split")
    add_subset_flags(p, "train on the first N train pairs (default: all)")
    p.add_argument("--eval-loss-pairs", "--eval-pairs", dest="eval_loss_pairs", type=int, default=500,
                   help="first N val pairs used for eval_loss during training")
    p.add_argument("--time-budget-min", type=float, help="stop training cleanly after this many minutes")
    p.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    p.add_argument("--no-direction", action="store_true", help="skip the fix-direction metric")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
