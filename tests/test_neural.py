import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bilstm import BiLSTM, build_vocab, code_tokens, parser as bilstm_parser, run as run_bilstm
from experiment_utils import classification_metrics, load_examples
from llm_utils import AnswerCollator, encode_candidate, predict_buggy


class TinyTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text == " fixed":
            return [3]
        if text == " buggy":
            return [4, 5]
        return [1] * len(text)


class ConstantLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(6, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask):
        logits = torch.tensor([0., 0., 0., 1., 2., 3.]).expand(*input_ids.shape, 6)
        return SimpleNamespace(logits=logits)


def test_token_order_indentation_and_incomplete_code():
    assert code_tokens("a - b") != code_tokens("b - a")
    assert "<INDENT>" in code_tokens("if x:\n    y()\n")
    assert code_tokens("f(")
    vocab = build_vocab(["training_name = 1"])
    assert "validation_only_name" not in vocab


def test_bilstm_ignores_right_padding():
    torch.manual_seed(42)
    model = BiLSTM(10, 4, 3, dropout=0).eval()
    short = model(torch.tensor([[2, 3]]), torch.tensor([2]))
    padded = model(torch.tensor([[2, 3, 0, 0]]), torch.tensor([2]))
    torch.testing.assert_close(short, padded)


def test_answer_mask_truncation_and_padding():
    tokenizer = TinyTokenizer()
    fixed, truncated = encode_candidate(tokenizer, "x" * 1000, "fixed", 64)
    buggy, _ = encode_candidate(tokenizer, "x", "buggy", 64)
    assert truncated and len(fixed["input_ids"]) == 63
    assert [x for x in fixed["labels"] if x != -100] == [3]
    assert [x for x in buggy["labels"] if x != -100] == [4, 5]
    same_code, _ = encode_candidate(tokenizer, "x" * 1000, "buggy", 64)
    assert fixed["input_ids"][:-1] == same_code["input_ids"][:-2]
    batch = AnswerCollator(0)([fixed, buggy])
    assert batch["labels"][0, -1] == -100
    assert batch["attention_mask"][0, -1] == 0


def test_likelihood_scores_all_answer_tokens():
    probabilities, truncated = predict_buggy(ConstantLM(), TinyTokenizer(), ["x", "y"], 512, 2)
    logp = F.log_softmax(torch.tensor([0., 0., 0., 1., 2., 3.]), dim=0)
    expected = torch.softmax(torch.stack([logp[3], logp[4] + logp[5]]), dim=0)[1].item()
    assert probabilities == pytest.approx([expected, expected])
    assert truncated == 0


def test_metrics_and_pair_sampling(tmp_path):
    pairs = pd.DataFrame({"buggy": ["a", "b", "c"], "fixed": ["d", "e", "f"]})
    pairs.to_parquet(tmp_path / "val.parquet")
    examples = load_examples(tmp_path, "val", limit_pairs=2)
    assert len(examples) == 4
    assert examples.groupby("pair").label.apply(set).tolist() == [{0, 1}, {0, 1}]
    metrics = classification_metrics([0, 1], [0.1, 0.9])
    assert metrics["f1"] == 1
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    with pytest.raises(FileNotFoundError, match="01_data"):
        load_examples(tmp_path, "train")


def test_bilstm_training_and_checkpoint(tmp_path):
    pairs = pd.DataFrame({"buggy": ["x = a - b", "f("], "fixed": ["x = b - a", "f()"]})
    for split in ("train", "val"):
        pairs.to_parquet(tmp_path / f"{split}.parquet")
    output = tmp_path / "model"
    args = bilstm_parser().parse_args([
        "--data-dir", str(tmp_path), "--output", str(output), "--epochs", "1",
        "--embedding-dim", "4", "--hidden-dim", "3", "--batch-size", "2",
    ])
    report = run_bilstm(args)
    assert report["n_examples"] == 4
    assert (output / "best.pt").exists()
    assert (output / "predictions.csv").exists()


def test_tiny_llama_lora_training_and_reload(tmp_path):
    pytest.importorskip("peft")
    from peft import PeftModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from llm_utils import load_model
    from train_lora import parser, run
    from safetensors.torch import load_file

    base = tmp_path / "tiny_base"
    backend = Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "</s>": 2, "buggy": 3, "fixed": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>", eos_token="</s>")
    tokenizer.save_pretrained(base)
    LlamaForCausalLM(LlamaConfig(vocab_size=5, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                               max_position_embeddings=128)).save_pretrained(base)
    pairs = pd.DataFrame({"buggy": ["a - b"], "fixed": ["b - a"]})
    for split in ("train", "val"):
        pairs.to_parquet(tmp_path / f"{split}.parquet")
    output = tmp_path / "lora"
    args = parser().parse_args(["--model", str(base), "--data-dir", str(tmp_path),
                               "--output", str(output), "--epochs", "1", "--batch-size", "2",
                               "--accumulation-steps", "1", "--max-length", "64"])
    report = run(args)
    assert report["n_examples"] == 2
    weights = load_file(output / "adapter" / "adapter_model.safetensors")
    assert any(tensor.abs().sum() > 0 for name, tensor in weights.items() if "lora_B" in name)
    model, tokenizer = load_model(str(base))
    model = PeftModel.from_pretrained(model, output / "adapter")
    probabilities, _ = predict_buggy(model, tokenizer, ["a - b"], 64)
    assert 0 <= probabilities[0] <= 1
    from llm_baseline import parser as eval_parser, run as evaluate
    recheck = evaluate(eval_parser().parse_args([
        "--model", str(base), "--adapter", str(output / "adapter"),
        "--data-dir", str(tmp_path), "--output", str(tmp_path / "recheck"), "--max-length", "64",
    ]))
    assert recheck["f1"] == report["f1"]
