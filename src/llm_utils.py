"""The same prompt and answer likelihood are used before and after LoRA."""

import torch
from torch.nn import functional as F


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


def load_model(model_id=MODEL_ID, quantized=False, revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if quantized and not torch.cuda.is_available():
        raise RuntimeError("4-bit mode needs a CUDA GPU. Use Colab T4 or omit --quantized.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"revision": revision}
    if quantized:
        kwargs.update(device_map={"": torch.cuda.current_device()}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16))
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if not quantized:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


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
    device = model.get_input_embeddings().weight.device
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
