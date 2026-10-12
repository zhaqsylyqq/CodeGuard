from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codeguard.config import DataConfig
from codeguard.data.pipeline import load_pairs, prepare
from codeguard.data.quality import code_hash
from codeguard.data.split import (
    SPLITS,
    assign_splits,
    drop_cross_split_duplicates,
    leakage_report,
    sample_pairs,
)
from codeguard.data.views import to_classification, to_generation

FRACTIONS = {"val": 0.1, "test": 0.1}


def make_pool(n_groups: int = 200, seed: int = 0) -> pd.DataFrame:
    """Pairs with very uneven group sizes, like real repositories."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 20, n_groups)
    projects = np.repeat([f"p{g}" for g in range(n_groups)], sizes)
    n = len(projects)
    buggy = [f"x = f({i})" for i in range(n)]
    fixed = [f"x = g({i})" for i in range(n)]
    return pd.DataFrame(
        {
            "pair_id": [f"id{i}" for i in range(n)],
            "project": projects,
            "buggy": buggy,
            "fixed": fixed,
            "added_lines": fixed,
            "sstub_pattern": "SINGLE_STMT",
            "is_py2": False,
            "buggy_hash": [code_hash(c) for c in buggy],
            "fixed_hash": [code_hash(c) for c in fixed],
        }
    )


@pytest.fixture
def split_pool() -> pd.DataFrame:
    df = make_pool()
    df["split"] = assign_splits(df["project"], 0.1, 0.1, seed=42)
    return df


def test_no_project_in_two_splits(split_pool):
    report = leakage_report(split_pool)
    assert all(v == 0 for v in report["shared_groups"].values())
    assert report["pairs_in_multiple_splits"] == 0
    assert set(split_pool["split"]) == set(SPLITS)


def test_split_is_deterministic_and_seed_dependent():
    groups = make_pool()["project"]
    a = assign_splits(groups, 0.1, 0.1, seed=42)
    b = assign_splits(groups, 0.1, 0.1, seed=42)
    c = assign_splits(groups, 0.1, 0.1, seed=7)
    assert a.equals(b)
    assert not a.equals(c)


def test_group_fractions_match_config(split_pool):
    groups = split_pool.groupby("split")["project"].nunique()
    assert groups["test"] == 20 and groups["val"] == 20 and groups["train"] == 160


def test_cross_split_duplicates_are_dropped():
    df = pd.DataFrame(
        {
            "split": ["train", "val", "test", "test"],
            "buggy_hash": ["A", "B", "C", "X"],
            "fixed_hash": ["a", "A", "b", "Y"],  # val shares "A" with train
        }
    )
    df.loc[2, "buggy_hash"] = "B"  # test shares "B" with val
    kept, dropped = drop_cross_split_duplicates(df)
    assert dropped == {"val": 1, "test": 0}  # val row dropped first, so test "B" survives
    assert list(kept.index) == [0, 2, 3]


def test_test_pair_overlapping_train_is_dropped():
    df = pd.DataFrame(
        {"split": ["train", "test"], "buggy_hash": ["A", "Z"], "fixed_hash": ["a", "A"]}
    )
    kept, dropped = drop_cross_split_duplicates(df)
    assert dropped["test"] == 1 and list(kept["split"]) == ["train"]


def test_samples_are_nested_across_sizes(split_pool):
    small, _ = sample_pairs(split_pool, 100, FRACTIONS, seed=42)
    large, _ = sample_pairs(split_pool, 400, FRACTIONS, seed=42)
    for split in SPLITS:
        s = set(small.loc[small["split"] == split, "pair_id"])
        big = set(large.loc[large["split"] == split, "pair_id"])
        assert s <= big
    assert small["split"].value_counts().to_dict() == {"train": 80, "val": 10, "test": 10}


def test_sample_respects_per_project_cap(split_pool):
    sample, _ = sample_pairs(split_pool, None, FRACTIONS, seed=42, max_per_group=2)
    assert sample.groupby("project").size().max() <= 2


def test_sample_warns_when_a_split_is_too_small(split_pool):
    _, warnings = sample_pairs(split_pool, 10**6, FRACTIONS, seed=42)
    assert len(warnings) == 3


def test_classification_view_is_balanced_and_keeps_pairs_together(split_pool):
    pairs = split_pool.head(50)
    rows = to_classification(pairs)
    assert len(rows) == 100
    assert rows["label"].tolist() == [1, 0] * 50
    assert (rows["pair_id"].iloc[::2].to_numpy() == rows["pair_id"].iloc[1::2].to_numpy()).all()
    assert rows.loc[0, "code"] == pairs.iloc[0]["buggy"]
    assert rows.loc[1, "code"] == pairs.iloc[0]["fixed"]


def test_generation_view_maps_buggy_to_fixed(split_pool):
    gen = to_generation(split_pool.head(3))
    assert gen["source"].tolist() == split_pool.head(3)["buggy"].tolist()
    assert gen["target"].tolist() == split_pool.head(3)["fixed"].tolist()


# ----------------------------------------------------------- full pipeline


def _data_config(ctssb_dir: Path, out: Path, n_pairs: int | None) -> DataConfig:
    return DataConfig.model_validate(
        {
            "name": "synthetic",
            "tier": "smoke",
            "source": {"name": "ctssb", "path": str(ctssb_dir)},
            "sample": {"n_pairs": n_pairs},
            "output_dir": str(out),
            "cache_dir": str(out.parent / "_cache"),
        }
    )


def test_prepare_end_to_end_on_synthetic_data(ctssb_dir, tmp_path):
    manifest = prepare(_data_config(ctssb_dir, tmp_path / "full", None))
    funnel = manifest["funnel"]

    assert funnel["raw_records"] == 154
    assert funnel["in_function"] < funnel["raw_records"]
    assert funnel["single_hunk"] == funnel["in_function"] - 1  # multi-hunk record
    assert funnel["non_whitespace_change"] == funnel["single_hunk"] - 1  # whitespace-only
    assert funnel["deduplicated"] == funnel["non_whitespace_change"] - 1  # the fork

    leak = manifest["leakage"]["sample"]
    assert all(v == 0 for v in leak["shared_groups"].values())
    assert all(v == 0 for v in leak["cross_split_snippet_duplicates"].values())
    assert manifest["pool"]["python2_markers"] == {"print_statement": 1}

    frames = [load_pairs(tmp_path / "full", s) for s in SPLITS]
    assert sum(len(f) for f in frames) == funnel["sampled"]
    assert not frames[0]["buggy"].str.contains("@@").any()


def test_prepare_is_reproducible_and_nested(ctssb_dir, tmp_path):
    a = prepare(_data_config(ctssb_dir, tmp_path / "a", 60))
    b = prepare(_data_config(ctssb_dir, tmp_path / "b", 60))
    prepare(_data_config(ctssb_dir, tmp_path / "big", 100))
    for split in SPLITS:
        assert a["files"][split]["content_hash"] == b["files"][split]["content_hash"]
        small_ids = set(load_pairs(tmp_path / "a", split)["pair_id"])
        big_ids = set(load_pairs(tmp_path / "big", split)["pair_id"])
        assert small_ids <= big_ids
