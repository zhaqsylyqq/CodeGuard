"""Train one LoRA rank per run; compare ranks on validation F1."""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments

from experiment_utils import DEFAULT_DATA_DIR, load_examples, run_metadata, save_report, seed_everything
from llm_utils import (ANSWERS, MODEL_ID, AnswerCollator, TimeBudgetCallback, encode_candidate, load_model,
                       predict_buggy, require_single_gpu, resolve_dtype)


def training_items(examples, tokenizer, max_length):
    return [encode_candidate(tokenizer, row.text, ANSWERS[row.label], max_length)[0]
            for row in examples.itertuples()]


def run(args):
    require_single_gpu()
    seed_everything(args.seed)
    train = load_examples(args.data_dir, "train", args.limit_pairs, args.seed)
    val = load_examples(args.data_dir, "val", args.limit_pairs, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model, args.quantized, args.revision, args.dtype)
    model.config.use_cache = False
    if args.quantized:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=args.rank, lora_alpha=2 * args.rank,
        lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))
    model.print_trainable_parameters()
    dtype = resolve_dtype(args.dtype)
    on_gpu = torch.cuda.is_available()
    settings = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation_steps, learning_rate=args.lr,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        bf16=on_gpu and dtype == torch.bfloat16, fp16=on_gpu and dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, report_to="none", seed=args.seed, data_seed=args.seed,
        dataloader_pin_memory=on_gpu,
        remove_unused_columns=False, label_names=["labels"],
    )
    budget = TimeBudgetCallback(args.time_budget_min)  # this script never resumes: fresh budget
    trainer = Trainer(model=model, args=settings,
                      train_dataset=training_items(train, tokenizer, args.max_length),
                      eval_dataset=training_items(val, tokenizer, args.max_length),
                      data_collator=AnswerCollator(tokenizer.pad_token_id), callbacks=[budget])
    trainer.train()
    adapter = output / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    model.gradient_checkpointing_disable()
    probabilities, truncated = predict_buggy(model, tokenizer, val.text, args.max_length, args.batch_size)
    report = save_report(output, val, probabilities, {
        "task": "detection", "model": args.model, "revision": args.revision,
        "method": "qlora" if args.quantized else "lora", "rank": args.rank, "split": "val", "seed": args.seed, "settings": vars(args),
        "truncated_examples": truncated, "best_checkpoint": trainer.state.best_model_checkpoint,
        "history": trainer.state.log_history, "stopped_by_time_budget": budget.stopped_early,
        **run_metadata(args),
    })
    print(json.dumps(report, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_ID,
                   help="base or instruct model id (the classification prompt is plain text either way)")
    p.add_argument("--revision")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="data/processed/small or data/processed/full")
    p.add_argument("--output", required=True)
    p.add_argument("--rank", type=int, choices=[8, 16, 32], default=16)
    p.add_argument("--quantized", action="store_true")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto: bf16 on GPUs that support it, else fp16; fp32 on CPU")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit-pairs", type=int)
    p.add_argument("--time-budget-min", type=float, help="stop training cleanly after this many minutes")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
