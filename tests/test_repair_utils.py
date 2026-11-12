import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment_utils import bootstrap_ci, paired_bootstrap_diff
from repair_utils import (
    assistant_message,
    choose_shots,
    edited_lines,
    encode_example,
    encode_prompt,
    localization_chance,
    max_new_tokens,
    parse_output,
    repair_metrics,
    stop_token_id,
    summarize,
    target_text,
)
from tiny_models import make_model, make_tokenizer

BUGGY = "def f(x):\n    if x > 0:\n        return x - 1\n    return 0"
FIXED = "def f(x):\n    if x > 0:\n        return x + 1\n    return 0"


# ---------------------------------------------------------------- prompt and label masking


def _label_span(item):
    return [t for t in item["labels"] if t != -100]


def test_labels_cover_exactly_the_answer_and_end_of_turn():
    tokenizer = make_tokenizer()
    item = encode_example(tokenizer, BUGGY, FIXED)
    labels = item["labels"]
    n_prompt = len(encode_prompt(tokenizer, BUGGY))
    assert labels[:n_prompt] == [-100] * n_prompt
    assert all(t != -100 for t in labels[n_prompt:])
    assert item["input_ids"][n_prompt:] == labels[n_prompt:]
    assert len(item["input_ids"]) == len(labels) == len(item["attention_mask"])
    # Decoded label span is the answer plus the token generation stops on, nothing after it.
    assert tokenizer.decode(_label_span(item)) == assistant_message(FIXED) + "\n<eot>"
    assert tokenizer.decode(_label_span(item)) == target_text(tokenizer, BUGGY, FIXED)
    assert labels[-1] == stop_token_id(tokenizer) == tokenizer.eos_token_id


def test_few_shot_turns_are_masked_too():
    tokenizer = make_tokenizer()
    shots = [("a = 1", "a = 2"), ("b()", "c()")]
    item = encode_example(tokenizer, BUGGY, FIXED, shots)
    assert tokenizer.decode(_label_span(item)) == assistant_message(FIXED) + "\n<eot>"
    prompt = tokenizer.decode(encode_prompt(tokenizer, BUGGY, shots))
    assert prompt.count("### Instruction:") == 3 and prompt.endswith("### Response:\n")
    assert prompt.count(assistant_message("a = 2")) == 1


def test_real_deepseek_template_masking_and_stop_token():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-instruct", local_files_only=True)
    except OSError:
        pytest.skip("deepseek-coder-1.3b-instruct tokenizer is not in the local HF cache")
    item = encode_example(tokenizer, BUGGY, FIXED)
    span = _label_span(item)
    assert tokenizer.decode(span) == assistant_message(FIXED) + "\n<|EOT|>"
    # Generation stops on the tokenizer's EOS; the training target must end with that same token.
    assert span[-1] == stop_token_id(tokenizer) == tokenizer.convert_tokens_to_ids("<|EOT|>")
    assert span.count(stop_token_id(tokenizer)) == 1
    assert item["input_ids"].count(tokenizer.bos_token_id) == 1 and item["input_ids"][0] == tokenizer.bos_token_id


def test_max_new_tokens_rule():
    tokenizer = make_tokenizer()  # one token per character
    assert max_new_tokens(tokenizer, "x" * 10) == 47
    assert max_new_tokens(tokenizer, "x" * 1000) == 512


# ---------------------------------------------------------------- parsing


def test_parse_fenced_block_with_text_around_it():
    code, ok = parse_output("Here is the fix:\n```python\n    return x + 1\n```\nDone.\n```python\nother\n```")
    assert ok and code == "    return x + 1"


def test_parse_keeps_indentation_and_strips_only_one_newline():
    code, ok = parse_output("```python\n\n    a = 1\n\n```")
    assert ok and code == "\n    a = 1\n"


def test_parse_without_fence_takes_raw_text():
    code, ok = parse_output("    x = 2\n")
    assert not ok and code == "    x = 2"


def test_parse_unclosed_fence_is_not_ok():
    code, ok = parse_output("```python\n    x = 2\n    y")
    assert not ok and code == "    x = 2\n    y"


def test_parse_fence_without_language_tag():
    assert parse_output("```\nx = 1\n```") == ("x = 1", True)


# ---------------------------------------------------------------- metrics


def test_edited_lines_replace_insert_delete_multiline():
    buggy = "a\nb\nc"
    assert edited_lines(buggy, "a\nB\nc") == {1}
    assert edited_lines(buggy, "a\nx\nb\nc") == {1}  # pure insertion touches the line it precedes
    assert edited_lines(buggy, "a\nb\nc\nd") == {3}  # insertion at the end
    assert edited_lines(buggy, "a\nc") == {1}  # pure deletion
    assert edited_lines(buggy, "A\nB\nc") == {0, 1}  # multi-line change
    assert edited_lines(buggy, "a\nb  \n\nc") == set()  # only whitespace / blank lines differ


def test_metrics_exact_fix():
    m = repair_metrics(BUGGY, FIXED, FIXED + "\n\n")
    assert m == {"exact_match": True, "changed": True, "localized": True, "localized_exact": True}


def test_metrics_copied_input():
    m = repair_metrics(BUGGY, FIXED, BUGGY)
    assert m == {"exact_match": False, "changed": False, "localized": False, "localized_exact": False}


def test_metrics_right_line_wrong_fix():
    wrong = BUGGY.replace("x - 1", "x * 1")
    assert repair_metrics(BUGGY, FIXED, wrong) == {
        "exact_match": False, "changed": True, "localized": True, "localized_exact": True}


def test_metrics_wrong_line_and_extra_line():
    wrong_line = BUGGY.replace("return 0", "return 1")
    assert repair_metrics(BUGGY, FIXED, wrong_line)["localized"] is False
    both = FIXED.replace("return 0", "return 1")  # right fix plus an extra edit
    m = repair_metrics(BUGGY, FIXED, both)
    assert m["localized"] and not m["localized_exact"] and not m["exact_match"]


def test_metrics_indentation_matters():
    buggy, fixed = "for i in x:\n    a()\nb()", "for i in x:\n    a()\n    b()"
    assert repair_metrics(buggy, fixed, fixed)["exact_match"]
    assert not repair_metrics(buggy, fixed, buggy)["changed"]


def test_localization_chance():
    assert localization_chance("a\nb\nc\nd", "a\nB\nc\nd") == 0.25
    assert localization_chance("a\nb", "a\nb\nc") == 0.0  # insertion after the last line


def _predictions():
    rows = []
    for i in range(40):
        pattern = "CHANGE_BINARY_OPERATOR" if i < 32 else f"RARE_{i % 2}"
        exact = i % 2 == 0
        rows.append({"pair_id": i, "sstub_pattern": pattern, "is_py2": i % 5 == 0, "buggy": BUGGY,
                     "fixed": FIXED, "prediction": FIXED if exact else BUGGY, "parse_ok": True,
                     "direction_correct": i % 4 != 0, **repair_metrics(BUGGY, FIXED, FIXED if exact else BUGGY)})
    return pd.DataFrame(rows)


def test_summarize_breakdowns_and_reference_points():
    pytest.importorskip("sacrebleu")
    summary = summarize(_predictions())
    overall = summary["overall"]
    assert overall["exact_match"] == 0.5 and overall["n"] == 40
    assert overall["direction_correct"] == 0.75 and overall["direction_chance"] == 0.5
    assert overall["copy_exact_match"] == 0.0
    assert overall["bleu"] > overall["copy_bleu"]
    assert set(summary["by_pattern"]) == {"CHANGE_BINARY_OPERATOR", "OTHER"}
    assert summary["by_pattern"]["OTHER"]["n"] == 8
    assert set(summary["by_py2"]) == {"true", "false"}
    low, high = summary["intervals"]["exact_match"]["low"], summary["intervals"]["exact_match"]["high"]
    assert low < 0.5 < high


def test_choose_shots_is_deterministic_and_diverse():
    train = pd.DataFrame({"pair_id": range(20), "buggy": ["a = 1"] * 19 + ["x\n" * 10],
                          "fixed": ["a = 2"] * 20, "sstub_pattern": [f"P{i % 4}" for i in range(20)]})
    shots = choose_shots(train, k=3, seed=42)
    assert len(shots) == 3 and shots["sstub_pattern"].nunique() == 3
    assert 19 not in shots["pair_id"].tolist()  # too long
    assert shots["pair_id"].tolist() == choose_shots(train, k=3, seed=42)["pair_id"].tolist()


# ---------------------------------------------------------------- bootstrap


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    values = np.r_[np.ones(30), np.zeros(70)]
    first, second = bootstrap_ci(values), bootstrap_ci(values)
    assert first == second
    assert first["mean"] == pytest.approx(0.3)
    assert first["low"] < 0.3 < first["high"]
    assert bootstrap_ci(np.zeros(5)) == {"mean": 0.0, "low": 0.0, "high": 0.0}


def test_paired_bootstrap_difference():
    a, b = np.r_[np.ones(60), np.zeros(40)], np.r_[np.ones(30), np.zeros(70)]
    diff = paired_bootstrap_diff(a, b)
    assert diff["mean"] == pytest.approx(0.3) and diff["low"] > 0
    with pytest.raises(ValueError):
        paired_bootstrap_diff(a, b[:10])


# ---------------------------------------------------------------- model calls


def test_direction_scoring_on_constant_lm():
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F

    from repair_utils import direction_scores

    tokenizer = make_tokenizer()
    vocab = len(tokenizer)
    base = torch.linspace(0, 1, vocab)

    class ConstantLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab, 2)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=base.expand(*input_ids.shape, vocab))

    logp = F.log_softmax(base, dim=0)
    buggies, fixeds = ["a = 1", "xx"], ["a = 22", "y"]
    s_fwd, s_bwd = direction_scores(ConstantLM(), tokenizer, buggies, fixeds, batch_size=2)
    for b, f, fwd, bwd in zip(buggies, fixeds, s_fwd, s_bwd):
        expect_fwd = float(sum(logp[t] for t in _label_span(encode_example(tokenizer, b, f))))
        expect_bwd = float(sum(logp[t] for t in _label_span(encode_example(tokenizer, f, b))))
        assert fwd == pytest.approx(expect_fwd, rel=1e-5) and bwd == pytest.approx(expect_bwd, rel=1e-5)
    # Under a constant LM the longer target always has the lower score.
    assert s_fwd[0] < s_bwd[0] and s_fwd[1] > s_bwd[1]


def test_left_padded_batch_generation_matches_one_by_one():
    pytest.importorskip("torch")
    from repair_utils import generate_repairs

    tokenizer = make_tokenizer()
    model = make_model(tokenizer)
    buggies = ["a = 1", "def g():\n    return [1, 2, 3]", "x"]
    batched = generate_repairs(model, tokenizer, buggies, batch_size=3)
    single = [generate_repairs(model, tokenizer, [b], batch_size=1)[0] for b in buggies]
    assert batched == single
    for buggy, (_, n_new) in zip(buggies, batched):
        assert 1 <= n_new <= max_new_tokens(tokenizer, buggy)
