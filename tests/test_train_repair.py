"""End-to-end smoke tests of train_repair.py / eval_repair.py on a tiny random model (CPU or GPU)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("sacrebleu")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tiny_models import save_tiny_model

PREDICTION_COLUMNS = {"pair_id", "sstub_pattern", "is_py2", "prediction", "parse_ok", "exact_match",
                      "localized", "localized_exact", "changed", "direction_correct", "n_new_tokens"}


def _pairs(n, offset=0):
    rows = []
    for i in range(offset, offset + n):
        rows.append({"pair_id": i, "sstub_pattern": f"P{i % 4}", "is_py2": i % 3 == 0,
                     "buggy": f"def f{i}(x):\n    return x - {i}", "fixed": f"def f{i}(x):\n    return x + {i}"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("repair")
    save_tiny_model(root / "tiny")
    data = root / "data"
    data.mkdir()
    for split, (n, offset) in {"train": (8, 0), "val": (4, 100), "test": (4, 200)}.items():
        _pairs(n, offset).to_parquet(data / f"{split}.parquet")
    return root


def _train_args(workspace, output, *extra):
    from train_repair import parser

    return parser().parse_args([
        "--model", str(workspace / "tiny"), "--data-dir", str(workspace / "data"), "--output", str(output),
        "--epochs", "1", "--batch-size", "2", "--accumulation-steps", "1", "--eval-batch-size", "4",
        "--max-length", "400", "--dtype", "fp32", *extra])


def test_train_writes_metrics_predictions_and_adapter(workspace):
    from train_repair import run

    output = workspace / "runs" / "lora_tiny"
    report = run(_train_args(workspace, output))
    assert (output / "adapter" / "adapter_config.json").exists()
    for split in ("val", "test"):
        table = pd.read_parquet(output / f"{split}_predictions.parquet")
        assert PREDICTION_COLUMNS <= set(table.columns) and len(table) == 4
        assert report["splits"][split]["overall"]["n"] == 4
    saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert saved["method"] == "lora" and saved["rank"] == 16 and saved["task"] == "repair"
    assert saved["global_step"] == 4 and saved["train_examples"] == 8
    assert 0 < saved["trainable_parameters"] < saved["total_parameters"]
    assert saved["adapter_bytes"] > 0 and saved["log_history"]
    assert saved["git_commit"] is None or len(saved["git_commit"]) == 40
    assert any("eval_loss" in row for row in saved["log_history"])
    assert "direction_correct" in saved["splits"]["val"]["overall"]
    # LoRA B matrices start at zero; after training at least one must have moved.
    from safetensors.torch import load_file
    weights = load_file(output / "adapter" / "adapter_model.safetensors")
    assert any(t.abs().sum() > 0 for name, t in weights.items() if "lora_B" in name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_fp16_training_on_gpu_updates_lora(workspace):
    # The Colab T4 has no native bf16, so it trains in fp16: LoRA weights must still move. The fp16
    # gradient scaler skips the first steps while it lowers its scale, so train for 12 steps.
    from safetensors.torch import load_file

    from train_repair import parser, run

    output = workspace / "runs" / "lora_fp16"
    args = parser().parse_args([
        "--model", str(workspace / "tiny"), "--data-dir", str(workspace / "data"), "--output", str(output),
        "--epochs", "3", "--batch-size", "2", "--accumulation-steps", "1", "--max-length", "400",
        "--dtype", "fp16", "--splits", "val", "--no-direction", "--limit-pairs", "8"])
    report = run(args)
    assert report["global_step"] == 12
    weights = load_file(output / "adapter" / "adapter_model.safetensors")
    assert any(t.abs().sum() > 0 for name, t in weights.items() if "lora_B" in name)


def test_second_run_is_a_no_op(workspace, monkeypatch):
    import train_repair

    output = workspace / "runs" / "lora_tiny"
    before = (output / "metrics.json").read_text(encoding="utf-8")
    monkeypatch.setattr(train_repair, "train", lambda *a: pytest.fail("must not train again"))
    train_repair.run(_train_args(workspace, output))
    assert (output / "metrics.json").read_text(encoding="utf-8") == before


def test_resume_from_checkpoint(workspace):
    from train_repair import run

    output = workspace / "runs" / "lora_resume"
    run(_train_args(workspace, output, "--splits", "val", "--no-direction"))
    # Simulate a crash after the checkpoints were written: only checkpoints remain.
    for name in ("metrics.json", "train_info.json", "val_predictions.parquet"):
        (output / name).unlink()
    report = run(_train_args(workspace, output, "--splits", "val", "--no-direction", "--epochs", "2"))
    assert report["resumed_from"].replace("\\", "/").endswith("checkpoint-4")
    assert report["global_step"] == 8


def test_eval_only_after_adapter_exists(workspace, monkeypatch):
    import train_repair

    output = workspace / "runs" / "lora_tiny"
    copy = workspace / "runs" / "lora_eval_only"
    import shutil
    shutil.copytree(output, copy)
    (copy / "metrics.json").unlink()
    (copy / "test_predictions.parquet").unlink()
    monkeypatch.setattr(train_repair, "train", lambda *a: pytest.fail("must not train again"))
    report = train_repair.run(_train_args(workspace, copy))
    assert report["splits"]["val"]["resumed"] and (copy / "test_predictions.parquet").exists()


def test_time_budget_stops_training_cleanly(workspace):
    from train_repair import run

    output = workspace / "runs" / "lora_budget"
    report = run(_train_args(workspace, output, "--time-budget-min", "0", "--splits", "val"))
    assert report["stopped_by_time_budget"] and report["global_step"] == 1
    assert (output / "adapter" / "adapter_config.json").exists() and (output / "metrics.json").exists()
    assert report["best_checkpoint"] is not None  # the stopped state was evaluated and saved


def test_refuses_more_than_one_gpu(workspace, monkeypatch):
    from train_repair import run

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
        run(_train_args(workspace, workspace / "runs" / "never"))
    assert not (workspace / "runs" / "never").exists()


@pytest.mark.parametrize("shots", ["0", "3"])
def test_eval_repair_zero_and_few_shot(workspace, shots):
    from eval_repair import parser, run

    output = workspace / "runs" / f"shots_{shots}"
    report = run(parser().parse_args([
        "--model", str(workspace / "tiny"), "--data-dir", str(workspace / "data"), "--output", str(output),
        "--shots", shots, "--batch-size", "3", "--dtype", "fp32", "--limit-pairs", "3"]))
    assert report["method"] == ("zero-shot" if shots == "0" else "3-shot")
    assert len(report["shot_pair_ids"]) == int(shots)
    for split in ("val", "test"):
        table = pd.read_parquet(output / f"{split}_predictions.parquet")
        assert PREDICTION_COLUMNS <= set(table.columns) and len(table) == 3
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8"))["splits"]["test"]["overall"]["n"] == 3


def test_eval_with_adapter_reproduces_training_eval(workspace):
    from eval_repair import parser, run

    trained = workspace / "runs" / "lora_tiny"
    report = run(parser().parse_args([
        "--model", str(workspace / "tiny"), "--adapter", str(trained / "adapter"),
        "--data-dir", str(workspace / "data"), "--output", str(workspace / "runs" / "recheck"),
        "--splits", "val", "--batch-size", "1", "--dtype", "fp32"]))
    original = pd.read_parquet(trained / "val_predictions.parquet")
    again = pd.read_parquet(workspace / "runs" / "recheck" / "val_predictions.parquet")
    assert report["method"] == "lora"
    assert again["prediction"].tolist() == original["prediction"].tolist()
