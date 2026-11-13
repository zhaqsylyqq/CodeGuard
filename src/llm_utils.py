"""Model loading shared by all LLM scripts, plus the legacy detection prompt and scoring.

The classification prompt and answer likelihood (`predict_buggy`) are used before and after LoRA,
so zero-shot and fine-tuned detection stay comparable. Do not change their behaviour.
"""

import time

import torch
from torch.nn import functional as F
from transformers import TrainerCallback


MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"
ANSWERS = ("fixed", "buggy")


def classification_prompt(code):
    return (
        "Classify the following Python snippet. It may contain a bug or be a fixed version.\n"
        "Answer with exactly one word: buggy or fixed.\n\n"
        f"Python code:\n{code}\n\nAnswer:"
    )


def encode_candidate(tokenizer, code, answer, max_length=512):
    prompt = tokenizer.encode(classification_prompt(code), add_special_tokens=False)
    target = tokenizer.encode(" " + answer, add_special_tokens=False)
    answer_budget = max(len(tokenizer.encode(" " + word, add_special_tokens=False)) for word in ANSWERS)
    if not target or max_length <= answer_budget + 1:
        raise ValueError("max_length is too small for the answer")
    budget = max_length - answer_budget
    truncated = len(prompt) > budget
    if truncated:
        # Keep instructions and the answer cue; both ends are needed for classification.
        head = min(48, budget // 2)
        prompt = prompt[:head] + prompt[-(budget - head):]
    return {"input_ids": prompt + target, "attention_mask": [1] * (len(prompt) + len(target)),
            "labels": [-100] * len(prompt) + target}, truncated


# ---------------------------------------------------------------- devices and loading


def resolve_dtype(dtype="auto"):
    """'auto' means bf16 on a CUDA GPU with native bf16 (Ampere or newer), fp16 on older GPUs such as
    the Colab T4, fp32 on CPU. Emulated bf16 is not counted: it is slow and Trainer may reject it."""
    if dtype != "auto":
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def require_single_gpu():
    """Training uses exactly one GPU. With several visible, Trainer would silently use all of them."""
    count = torch.cuda.device_count()
    if count > 1:
        raise SystemExit(
            f"{count} GPUs are visible, but training must run on exactly one. "
            "Select one, for example: CUDA_VISIBLE_DEVICES=0 python src/train_repair.py ..."
        )


def load_model(model_id=MODEL_ID, quantized=False, revision=None, dtype="auto", multi_gpu=False):
    """Load a causal LM and its tokenizer.

    quantized: 4-bit NF4 weights with `dtype` compute (QLoRA). multi_gpu: spread the layers over
    all visible GPUs with device_map="auto" (only for evaluating models too big for one card).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if (quantized or multi_gpu) and not torch.cuda.is_available():
        raise RuntimeError("4-bit and multi-GPU modes need a CUDA GPU. Omit --quantized / --multi-gpu.")
    compute_dtype = resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"revision": revision, "dtype": compute_dtype}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype)
    if multi_gpu:
        kwargs["device_map"] = "auto"
    elif quantized:
        kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not quantized and not multi_gpu:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def model_device(model):
    """Device of the input embeddings: where input ids must be placed."""
    return model.get_input_embeddings().weight.device


class TimeBudgetCallback(TrainerCallback):
    """Stop training cleanly after `minutes`; the script then still saves and evaluates."""

    def __init__(self, minutes):
        self.seconds = None if minutes is None else minutes * 60
        self.start = None
        self.stopped_early = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.seconds is not None and time.monotonic() - self.start >= self.seconds:
            control.should_training_stop = True
            # Evaluate and save the state reached, so it can still be picked as the best checkpoint.
            control.should_evaluate = args.eval_strategy != "no"
            control.should_save = args.save_strategy != "no"
            self.stopped_early = True
        return control


# ---------------------------------------------------------------- legacy detection scoring


class AnswerCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, items):
        length = max(len(item["input_ids"]) for item in items)
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for item in items:
            padding = length - len(item["input_ids"])
            for key, fill in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                batch[key].append(item[key] + [fill] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


@torch.inference_mode()
def predict_buggy(model, tokenizer, texts, max_length=512, batch_size=4):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    collator = AnswerCollator(tokenizer.pad_token_id)
    device = model_device(model)
    probabilities, truncated_count = [], 0
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        items = []
        for text in texts[start:start + batch_size]:
            candidates = [encode_candidate(tokenizer, text, answer, max_length) for answer in ANSWERS]
            truncated_count += int(any(truncated for _, truncated in candidates))
            items.extend(item for item, _ in candidates)
        batch = {key: value.to(device) for key, value in collator(items).items()}
        labels = batch.pop("labels")[:, 1:]
        logits = model(**batch).logits[:, :-1].float()
        mask = labels != -100
        token_scores = F.log_softmax(logits, dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        scores = (token_scores * mask).sum(dim=1).reshape(-1, 2)
        probabilities.extend(scores.softmax(dim=1)[:, 1].cpu().tolist())
    return probabilities, truncated_count
