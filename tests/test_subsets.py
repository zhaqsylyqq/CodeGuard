"""--train-pairs / --eval-pairs-val / --eval-pairs-test take deterministic prefixes of each split."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_utils import load_pairs


def _write(tmp_path, n=10):
    pd.DataFrame({"pair_id": range(100, 100 + n), "buggy": [f"a = {i}" for i in range(n)],
                  "fixed": [f"a = {i + 1}" for i in range(n)]}).to_parquet(tmp_path / "val.parquet")


def test_first_is_a_prefix_and_smaller_prefixes_are_subsets(tmp_path):
    _write(tmp_path)
    three, seven = load_pairs(tmp_path, "val", first=3), load_pairs(tmp_path, "val", first=7)
    assert three["pair_id"].tolist() == [100, 101, 102]
    assert seven["pair_id"].tolist()[:3] == three["pair_id"].tolist()
    assert len(load_pairs(tmp_path, "val", first=50)) == 10
    with pytest.raises(ValueError):
        load_pairs(tmp_path, "val", first=0)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_stored_small_tier_is_a_prefix_of_the_full_tier(split):
    small, full = ROOT / "data/processed/small", ROOT / "data/processed/full"
    if not (small / f"{split}.parquet").exists() or not (full / f"{split}.parquet").exists():
        pytest.skip("processed data tiers are not present")
    small_ids = load_pairs(small, split)["pair_id"].tolist()
    assert load_pairs(full, split, first=len(small_ids))["pair_id"].tolist() == small_ids


def test_scripts_train_and_evaluate_on_prefixes(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("sacrebleu")
    sys.path.insert(0, str(ROOT / "tests"))
    from test_train_repair import _pairs
    from tiny_models import save_tiny_model

    save_tiny_model(tmp_path / "tiny")
    data = tmp_path / "data"
    data.mkdir()
    for split, (n, offset) in {"train": (8, 0), "val": (6, 100), "test": (6, 200)}.items():
        _pairs(n, offset).to_parquet(data / f"{split}.parquet")
    common = ["--model", str(tmp_path / "tiny"), "--data-dir", str(data), "--dtype", "fp32",
              "--eval-pairs-val", "2", "--eval-pairs-test", "3"]

    from train_repair import parser as train_parser, run as train_run
    report = train_run(train_parser().parse_args(common + [
        "--output", str(tmp_path / "lora"), "--train-pairs", "4", "--epochs", "1", "--batch-size", "2",
        "--accumulation-steps", "1", "--eval-batch-size", "3", "--max-length", "400"]))
    assert report["train_pairs"] == 4 and report["train_examples_seen"] == 4
    from eval_repair import parser as eval_parser, run as eval_run
    eval_run(eval_parser().parse_args(common + ["--output", str(tmp_path / "zs"), "--batch-size", "3",
                                                "--shots", "3", "--train-pairs", "6"]))
    for run in ("lora", "zs"):
        for split, n, offset in (("val", 2, 100), ("test", 3, 200)):
            table = pd.read_parquet(tmp_path / run / f"{split}_predictions.parquet")
            assert table["pair_id"].tolist() == list(range(offset, offset + n))
    shots = json.loads((tmp_path / "zs" / "metrics.json").read_text(encoding="utf-8"))["shot_pair_ids"]
    assert all(pid < 6 for pid in shots)  # chosen from the first 6 train pairs only


def test_extending_a_prediction_prefix_equals_evaluating_all_at_once(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sacrebleu")
    sys.path.insert(0, str(ROOT / "tests"))
    from eval_repair import parser, run
    from test_train_repair import _pairs
    from tiny_models import save_tiny_model

    save_tiny_model(tmp_path / "tiny")
    for split, (n, offset) in {"train": (8, 0), "val": (6, 100), "test": (6, 200)}.items():
        _pairs(n, offset).to_parquet(tmp_path / f"{split}.parquet")
    base = ["--model", str(tmp_path / "tiny"), "--data-dir", str(tmp_path), "--dtype", "fp32", "--batch-size", "4"]
    run(parser().parse_args(base + ["--output", str(tmp_path / "all"), "--eval-pairs-test", "5"]))
    run(parser().parse_args(base + ["--output", str(tmp_path / "grow"), "--eval-pairs-test", "2"]))
    (tmp_path / "grow" / "metrics.json").unlink()
    report = run(parser().parse_args(base + ["--output", str(tmp_path / "grow"), "--eval-pairs-test", "5"]))
    whole = pd.read_parquet(tmp_path / "all" / "test_predictions.parquet")
    grown = pd.read_parquet(tmp_path / "grow" / "test_predictions.parquet")
    cols = ["pair_id", "prediction", "exact_match", "s_fwd", "s_bwd"]
    pd.testing.assert_frame_equal(whole[cols], grown[cols], check_exact=False, rtol=1e-4)
    assert report["splits"]["test"]["extended_to"] == 5 and report["splits"]["test"]["overall"]["n"] == 5
