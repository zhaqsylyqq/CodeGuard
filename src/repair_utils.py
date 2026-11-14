"""Bug repair: one prompt for every setting, label masking, generation, parsing and metrics.

Task: given the buggy snippet, produce the whole fixed snippet. The same prompt is used for
zero-shot, few-shot, LoRA training and the demo, rendered with the tokenizer's own chat template.
"""

import difflib

import numpy as np
import pandas as pd

from data_utils import normalize_code
from experiment_utils import bootstrap_ci

INSTRUCTION = (
    "Fix the bug in the following Python code. It is a fragment of a larger file and contains "
    "exactly one buggy statement. Return the complete corrected fragment in a single Python code "
    "block and nothing else."
)
FENCE = "```"
ANSWER_PREFIX = f"{FENCE}python\n"  # prefilled at generation: the model continues inside the code block
GENERATION_RULE = "prefill_fence_v2"  # recorded in metrics.json; runs without it used the old rule (v1)
MAX_NEW_TOKENS_CAP = 512
MIN_PATTERN_PAIRS = 30  # bug types with fewer pairs are grouped as OTHER in breakdowns


# ---------------------------------------------------------------- prompt and encoding


def user_message(buggy):
    return f"{INSTRUCTION}\n\n{FENCE}python\n{buggy}\n{FENCE}"


def assistant_message(fixed):
    return f"{FENCE}python\n{fixed}\n{FENCE}"


def build_messages(buggy, shots=()):
    """Chat messages for one item. `shots` are (buggy, fixed) pairs shown as earlier turns."""
    messages = []
    for shot_buggy, shot_fixed in shots:
        messages.append({"role": "user", "content": user_message(shot_buggy)})
        messages.append({"role": "assistant", "content": assistant_message(shot_fixed)})
    messages.append({"role": "user", "content": user_message(buggy)})
    return messages


def stop_token_id(tokenizer):
    """The token that ends the assistant turn. Generation stops on it and training learns it."""
    return tokenizer.eos_token_id


def token_ids(tokenizer, text):
    # The chat template already writes special tokens (such as BOS) as text, so add none here.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def prompt_text(tokenizer, buggy, shots=()):
    return tokenizer.apply_chat_template(build_messages(buggy, shots), tokenize=False,
                                         add_generation_prompt=True)


def target_text(tokenizer, buggy, fixed, shots=()):
    """What the model must write after the prompt: the answer up to and including end-of-turn.

    Taken from the chat template itself (full conversation minus the prompt), so every model gets
    its native format. Anything the template adds after the end-of-turn token is dropped, because
    generation stops there.
    """
    prompt = prompt_text(tokenizer, buggy, shots)
    answer = assistant_message(fixed)
    messages = build_messages(buggy, shots) + [{"role": "assistant", "content": answer}]
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("Chat template: the generation prompt is not a prefix of the full conversation")
    target = full[len(prompt):]
    answer_end = target.find(answer) + len(answer)
    if answer_end < len(answer):
        raise ValueError("Chat template: the answer text is not in the rendered conversation")
    eos = tokenizer.eos_token
    end = target.find(eos, answer_end)
    return target + eos if end == -1 else target[:end + len(eos)]


def encode_prompt(tokenizer, buggy, shots=()):
    return token_ids(tokenizer, prompt_text(tokenizer, buggy, shots))


def encode_example(tokenizer, buggy, fixed, shots=()):
    """Training / scoring item: labels cover only the answer and end-of-turn; prompt is -100."""
    prompt = encode_prompt(tokenizer, buggy, shots)
    target = token_ids(tokenizer, target_text(tokenizer, buggy, fixed, shots))
    return {"input_ids": prompt + target, "attention_mask": [1] * (len(prompt) + len(target)),
            "labels": [-100] * len(prompt) + target}


def answer_lead(tokenizer, buggy, shots=()):
    """What the chat template writes before the answer, plus the opening code fence.

    Generation is prefilled with this text, so every setting (zero-shot, few-shot, LoRA) starts
    writing code at once instead of spending its token budget on an explanation first.
    """
    probe = "pass"
    target = target_text(tokenizer, buggy, probe, shots)
    return target[:target.find(assistant_message(probe))] + ANSWER_PREFIX


def encode_generation_prompt(tokenizer, buggy, shots=()):
    """Prompt tokens plus the prefilled answer lead, tokenized as in training (prompt and answer apart)."""
    return encode_prompt(tokenizer, buggy, shots) + token_ids(tokenizer, answer_lead(tokenizer, buggy, shots))


def check_prefill(tokenizer, buggy, fixed, shots=()):
    """The prefilled tokens must be exactly the first tokens of the training target; otherwise a LoRA
    model would see a token sequence at generation that it never saw in training."""
    lead = token_ids(tokenizer, answer_lead(tokenizer, buggy, shots))
    target = token_ids(tokenizer, target_text(tokenizer, buggy, fixed, shots))
    if target[:len(lead)] != lead:
        raise ValueError("Prefill mismatch: the training target does not start with the tokens of "
                         f"{answer_lead(tokenizer, buggy, shots)!r} for this tokenizer")


def max_new_tokens(tokenizer, buggy):
    """Generation budget: enough for the fenced fix, never more than 512 tokens."""
    return min(MAX_NEW_TOKENS_CAP, int(1.5 * len(token_ids(tokenizer, buggy)) + 32))


def choose_shots(train, k=3, seed=42, max_lines=6, max_chars=300):
    """k short train pairs with different bug types; the same ones are used for every test item."""
    short = train[(train["buggy"].str.count("\n") < max_lines) & (train["fixed"].str.count("\n") < max_lines)
                  & (train["buggy"].str.len() <= max_chars) & (train["fixed"].str.len() <= max_chars)]
    chosen, patterns = [], set()
    for row in short.sample(frac=1.0, random_state=seed).itertuples():
        if row.sstub_pattern not in patterns:
            chosen.append(row)
            patterns.add(row.sstub_pattern)
        if len(chosen) == k:
            break
    if len(chosen) < k:
        raise ValueError(f"Could not find {k} short train pairs with different bug types")
    return pd.DataFrame(chosen).drop(columns="Index")


# ---------------------------------------------------------------- parsing


def parse_output(text, prefilled=True):
    """Code of the answer and whether its code block was closed.

    prefilled=True (current rule): `text` is what the model wrote after the prefilled opening fence,
    so the code runs up to the first fence, which closes the block; no fence means the token budget
    ended inside the block. prefilled=False (old v1 outputs): code from the first fenced block, or the
    raw text if there is none.

    Exactly one leading and one trailing newline are removed; leading spaces of the first line are
    kept because indentation is meaningful in Python.
    """
    if prefilled:
        end = text.find(FENCE)
        code, parse_ok = (text, False) if end == -1 else (text[:end], True)
        if code.startswith("\n"):
            code = code[1:]
        if code.endswith("\n"):
            code = code[:-1]
        return code, parse_ok
    start = text.find(FENCE)
    if start == -1:
        code, parse_ok = text, False
    else:
        body = text[start + len(FENCE):]
        newline = body.find("\n")
        body = "" if newline == -1 else body[newline:]  # skip the language tag on the fence line
        end = body.find(FENCE)
        parse_ok = end != -1
        code = body if end == -1 else body[:end]
    if code.startswith("\n"):
        code = code[1:]
    if code.endswith("\n"):
        code = code[:-1]
    return code, parse_ok


# ---------------------------------------------------------------- metrics


def edited_lines(buggy, other):
    """Indices of the buggy lines that `other` changes (normalized lines, difflib opcodes).

    A pure insertion counts as touching the line it is inserted before.
    """
    a = normalize_code(buggy).split("\n")
    b = normalize_code(other).split("\n")
    lines = set()
    for tag, i1, i2, _, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            lines.update(range(i1, max(i2, i1 + 1)))
    return lines


def repair_metrics(buggy, fixed, prediction):
    """Per-pair metrics of section 6.1 of the guide."""
    norm_prediction = normalize_code(prediction)
    changed = norm_prediction != normalize_code(buggy)
    developer, model = edited_lines(buggy, fixed), edited_lines(buggy, prediction)
    return {
        "exact_match": norm_prediction == normalize_code(fixed),
        "changed": changed,
        "localized": changed and bool(developer & model),
        "localized_exact": changed and developer == model,
    }


def localization_chance(buggy, fixed):
    """Chance that one random line edit touches a line the developer edited."""
    n_lines = len(normalize_code(buggy).split("\n"))
    developer = {line for line in edited_lines(buggy, fixed) if line < n_lines}
    return len(developer) / n_lines


def corpus_bleu(predictions, references):
    """Corpus BLEU on normalized code. Always report it next to the copy baseline."""
    import sacrebleu

    hypotheses = [normalize_code(p) for p in predictions]
    return float(sacrebleu.corpus_bleu(hypotheses, [[normalize_code(r) for r in references]]).score)


def _rates(frame):
    out = {"n": int(len(frame))}
    for column in ("exact_match", "changed", "localized", "localized_exact", "parse_ok"):
        out[column] = float(frame[column].mean())
    if "direction_correct" in frame and frame["direction_correct"].notna().all():
        out["direction_correct"] = float(frame["direction_correct"].astype(float).mean())
    return out


def summarize(predictions):
    """Metrics for one split: overall (with bootstrap intervals and reference points) and breakdowns."""
    overall = _rates(predictions)
    overall["bleu"] = corpus_bleu(predictions["prediction"], predictions["fixed"])
    overall["copy_bleu"] = corpus_bleu(predictions["buggy"], predictions["fixed"])
    overall["copy_exact_match"] = 0.0  # buggy != fixed for every pair by construction of the data
    overall["localization_chance"] = float(np.mean(
        [localization_chance(b, f) for b, f in zip(predictions["buggy"], predictions["fixed"])]))
    overall["direction_chance"] = 0.5
    intervals = {}
    for column in ("exact_match", "localized", "localized_exact", "direction_correct"):
        if column in overall:
            intervals[column] = bootstrap_ci(predictions[column].astype(float))
    counts = predictions["sstub_pattern"].value_counts()
    pattern = predictions["sstub_pattern"].where(
        predictions["sstub_pattern"].map(counts) >= MIN_PATTERN_PAIRS, "OTHER")
    return {
        "overall": overall,
        "intervals": intervals,
        "by_pattern": {name: _rates(group) for name, group in predictions.groupby(pattern)},
        "by_py2": {str(bool(name)).lower(): _rates(group)
                   for name, group in predictions.groupby(predictions["is_py2"].astype(bool))},
    }


# ---------------------------------------------------------------- model calls


def fence_stop_criteria(tokenizer, width):
    """Stops each sequence once its generated part contains a code fence (the block is closed)."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class FenceClosed(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            tails = input_ids[:, width:][:, -4:].tolist()  # a fence spans at most three tokens
            done = [FENCE in tokenizer.decode(tail, skip_special_tokens=True) for tail in tails]
            return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

    return StoppingCriteriaList([FenceClosed()])


def cut_at_fence(tokenizer, tokens):
    """Tokens up to and including the one that completes the first code fence."""
    for end in range(1, len(tokens) + 1):
        if FENCE in tokenizer.decode(tokens[max(0, end - 4):end], skip_special_tokens=True):
            return tokens[:end]
    return tokens


def generate_repairs(model, tokenizer, buggies, shots=(), batch_size=8):
    """Greedy, batched, left-padded generation. Returns [(raw_text, n_new_tokens)] in input order.

    The answer is prefilled with the opening code fence (`answer_lead`), so `raw_text` is what the
    model wrote after it. Each item stops when its code block is closed, at its end-of-turn token or
    at its own token budget (`max_new_tokens`), whichever comes first, so the result does not depend
    on which other items share its batch.
    """
    import torch

    from llm_utils import model_device

    model.eval()
    prompts = [encode_generation_prompt(tokenizer, buggy, shots) for buggy in buggies]
    budgets = [max_new_tokens(tokenizer, buggy) for buggy in buggies]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    pad, stop = tokenizer.pad_token_id, stop_token_id(tokenizer)
    device = model_device(model)
    results = [None] * len(prompts)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            width = max(len(prompts[i]) for i in batch)
            input_ids = torch.tensor([[pad] * (width - len(prompts[i])) + prompts[i] for i in batch])
            attention = torch.tensor([[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in batch])
            generated = model.generate(
                input_ids=input_ids.to(device), attention_mask=attention.to(device),
                max_new_tokens=max(budgets[i] for i in batch), do_sample=False,
                eos_token_id=stop, pad_token_id=pad, stopping_criteria=fence_stop_criteria(tokenizer, width))
            for row, i in enumerate(batch):
                tokens = generated[row, width:].tolist()[:budgets[i]]
                if stop in tokens:
                    tokens = tokens[:tokens.index(stop) + 1]
                tokens = cut_at_fence(tokenizer, tokens)
                results[i] = (tokenizer.decode(tokens, skip_special_tokens=True), len(tokens))
    return results


def sequence_logprobs(model, items, pad_token_id, batch_size=8):
    """Sum of log-probabilities of the labelled tokens of each item (teacher forcing)."""
    import torch
    from torch.nn import functional as F

    from llm_utils import AnswerCollator, model_device

    model.eval()
    collator = AnswerCollator(pad_token_id)
    device = model_device(model)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]), reverse=True)
    scores = [None] * len(items)
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            batch_index = order[start:start + batch_size]
            batch = {k: v.to(device) for k, v in collator([items[i] for i in batch_index]).items()}
            labels = batch.pop("labels")[:, 1:]
            logits = model(**batch).logits[:, :-1].float()
            mask = labels != -100
            token_scores = F.log_softmax(logits, dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            for row, i in enumerate(batch_index):
                scores[i] = float((token_scores[row] * mask[row]).sum())
    return scores


def direction_scores(model, tokenizer, buggies, fixeds, shots=(), batch_size=8):
    """s_fwd = log P(fixed | prompt(buggy)), s_bwd = log P(buggy | prompt(fixed))."""
    forward = [encode_example(tokenizer, b, f, shots) for b, f in zip(buggies, fixeds)]
    backward = [encode_example(tokenizer, f, b, shots) for b, f in zip(buggies, fixeds)]
    s_fwd = sequence_logprobs(model, forward, tokenizer.pad_token_id, batch_size)
    s_bwd = sequence_logprobs(model, backward, tokenizer.pad_token_id, batch_size)
    return s_fwd, s_bwd


def evaluate_split(model, tokenizer, pairs, shots=(), batch_size=8, direction=True):
    """Predictions table for one split (one row per pair) and its summary metrics."""
    buggies, fixeds = pairs["buggy"].tolist(), pairs["fixed"].tolist()
    for buggy, fixed in list(zip(buggies, fixeds))[:20]:
        check_prefill(tokenizer, buggy, fixed, shots)
    outputs = generate_repairs(model, tokenizer, buggies, shots, batch_size)
    rows = []
    for buggy, fixed, (raw, n_new) in zip(buggies, fixeds, outputs):
        prediction, parse_ok = parse_output(raw)
        rows.append({"prediction": prediction, "raw_output": raw, "parse_ok": parse_ok,
                     "n_new_tokens": n_new, **repair_metrics(buggy, fixed, prediction)})
    table = pd.DataFrame(rows)
    table.insert(0, "pair_id", pairs["pair_id"].to_numpy())
    table.insert(1, "sstub_pattern", pairs["sstub_pattern"].to_numpy())
    table.insert(2, "is_py2", pairs["is_py2"].to_numpy())
    table["buggy"], table["fixed"] = buggies, fixeds
    if direction:
        s_fwd, s_bwd = direction_scores(model, tokenizer, buggies, fixeds, shots, batch_size)
        table["s_fwd"], table["s_bwd"] = s_fwd, s_bwd
        table["direction_correct"] = table["s_fwd"] > table["s_bwd"]
    else:
        table["direction_correct"] = None
    return table, summarize(table)
