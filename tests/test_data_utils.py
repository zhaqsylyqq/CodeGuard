import sys
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import (
    check_leakage,
    code_hash,
    is_python2,
    load_pool,
    normalize_code,
    pairs_to_examples,
    rebuild_snippets,
    remove_cross_split_duplicates,
    sample_nested,
    split_by_project,
    within_length_limit,
)


@pytest.mark.parametrize("compressed", [False, True])
def test_load_pool_accepts_plain_and_compressed_jsonl(tmp_path, compressed):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [
        {"project": "demo", "file_path": "a.py", "likely_bug": True,
         "in_function": inside, "diff": "@@ -1 +1 @@\n-a\n+b\n", "sstub_pattern": "CHANGE"}
        for inside in (True, False)
    ]
    text = "\n".join(json.dumps(row) for row in rows)
    plain = raw / "file-0.jsonl"
    plain.write_text(text, encoding="utf-8")
    if compressed:
        with gzip.open(raw / "file-0.jsonl.gz", "wt", encoding="utf-8") as stream:
            stream.write(text)
    output = tmp_path / "processed" / "pool.parquet"
    pool, n_raw = load_pool(raw, output)
    assert n_raw == 2
    assert len(pool) == 1
    cached, cached_count = load_pool(raw, output)
    pd.testing.assert_frame_equal(cached, pool.reset_index(drop=True))
    assert cached_count == n_raw


def test_load_pool_missing_files_explains_expected_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Place the CTSSB files"):
        load_pool(tmp_path / "missing", tmp_path / "pool.parquet")

DIFF = (
    "@@ -1,4 +1,4 @@ header\n"
    " def f(x):\n"
    "-    return x + 1\n"
    "+    return x + 2\n"
    " \n"
    " y = 3\n"
    "\\ No newline at end of file\n"
)


def test_rebuild_snippets_splits_minus_plus_and_context():
    buggy, fixed, n_hunks = rebuild_snippets(DIFF)
    assert buggy == "def f(x):\n    return x + 1\n\ny = 3"
    assert fixed == "def f(x):\n    return x + 2\n\ny = 3"
    assert n_hunks == 1


def test_rebuild_snippets_counts_hunks():
    assert rebuild_snippets("@@ -1 +1 @@\n-a\n+b\n@@ -9 +9 @@\n-c\n+d\n")[2] == 2


def test_normalize_ignores_trailing_spaces_and_blank_lines():
    assert normalize_code("a = 1   \n\n\nb = 2\n") == "a = 1\nb = 2"


def test_indentation_only_fix_is_not_a_duplicate():
    # Moving a line into the loop body is a real fix in Python.
    buggy = "for i in x:\n    a()\nb()"
    fixed = "for i in x:\n    a()\n    b()"
    assert code_hash(buggy) != code_hash(fixed)


def test_trailing_whitespace_fix_is_a_duplicate():
    assert code_hash("a = 1  \n") == code_hash("a = 1\n\n")


def test_is_python2():
    assert is_python2('print "hello"')
    assert is_python2("except ValueError, e:")
    assert is_python2("x = 10L")
    assert not is_python2('print("hello")')


def _table(n_projects=40, rows_per_project=5):
    rows = []
    for p in range(n_projects):
        for r in range(rows_per_project):
            rows.append({"project": f"p{p}", "buggy_hash": f"b{p}-{r}", "fixed_hash": f"f{p}-{r}"})
    return pd.DataFrame(rows)


def test_split_keeps_projects_apart():
    df = _table()
    df["split"] = split_by_project(df, seed=42)
    assert set(df["split"]) == {"train", "val", "test"}
    assert df.groupby("project")["split"].nunique().max() == 1
    check_leakage(df)  # raises if a project is shared


def test_remove_cross_split_duplicates():
    df = _table(n_projects=3, rows_per_project=1)
    df["split"] = ["train", "val", "test"]
    df.loc[1, "buggy_hash"] = df.loc[0, "buggy_hash"]  # val repeats a train snippet
    df.loc[2, "fixed_hash"] = df.loc[1, "fixed_hash"]  # test repeats a val snippet
    clean, dropped = remove_cross_split_duplicates(df)
    assert dropped == {"val": 1, "test": 0}  # the dropped val row no longer counts
    assert list(clean["split"]) == ["train", "test"]


def test_pairs_to_examples_gives_balanced_labels():
    df = pd.DataFrame({"buggy": ["a = 1", "b = 2"], "fixed": ["a = 2", "b = 3"]})
    examples = pairs_to_examples(df)
    assert len(examples) == 4
    assert examples["label"].tolist() == [1, 1, 0, 0]
    assert examples.loc[examples["label"] == 1, "text"].tolist() == ["a = 1", "b = 2"]


NESTED_SIZES = {"small": {"train": 4, "val": 2, "test": 2}, "full": {"train": 10, "val": 5, "test": 6}}


def _split_table(n_per_split=(30, 10, 12)):
    rows = []
    for name, n in zip(["train", "val", "test"], n_per_split):
        for i in range(n):
            rows.append({"pair_id": f"{name}{i}", "split": name, "buggy": "a", "fixed": "b"})
    return pd.DataFrame(rows)


def test_sample_nested_small_is_subset_of_full_in_every_split():
    tiers = sample_nested(_split_table(), NESTED_SIZES, seed=42)
    for name in ["train", "val", "test"]:
        small = set(tiers["small"].loc[tiers["small"]["split"] == name, "pair_id"])
        full = set(tiers["full"].loc[tiers["full"]["split"] == name, "pair_id"])
        assert small < full


def test_sample_nested_sizes_prefixes_and_determinism():
    df = _split_table()
    tiers = sample_nested(df, NESTED_SIZES, seed=42)
    assert list(tiers) == ["small", "full"]
    for tier, sizes in NESTED_SIZES.items():
        out = tiers[tier]
        assert list(out.columns) == list(df.columns)
        assert list(out.index) == list(range(len(out)))
        assert out["split"].value_counts().to_dict() == sizes
        assert out["pair_id"].is_unique
    for name in ["train", "val", "test"]:
        small = tiers["small"].loc[tiers["small"]["split"] == name, "pair_id"].tolist()
        full = tiers["full"].loc[tiers["full"]["split"] == name, "pair_id"].tolist()
        assert full[: len(small)] == small  # small is a prefix of the same shuffled order
    again = sample_nested(df, NESTED_SIZES, seed=42)
    for tier in tiers:
        pd.testing.assert_frame_equal(tiers[tier], again[tier])
    other = sample_nested(df, NESTED_SIZES, seed=7)
    assert not tiers["full"]["pair_id"].equals(other["full"]["pair_id"])


def test_sample_nested_too_many_rows_raises():
    with pytest.raises(ValueError, match="val"):
        sample_nested(_split_table(), {"full": {"train": 5, "val": 11, "test": 1}}, seed=42)


def test_within_length_limit():
    at_limit_chars = "x" * 2000
    at_limit_lines = "\n".join(["a"] * 60)
    one_line_over = "\n".join(["a"] * 61)
    df = pd.DataFrame({
        "buggy": ["a = 1", at_limit_chars, at_limit_lines, "x" * 2001, one_line_over, "a"],
        "fixed": ["a = 2", at_limit_chars, at_limit_lines, "a", "a", "x" * 2001],
    })
    assert within_length_limit(df).tolist() == [True, True, True, False, False, False]
    # custom limits: 3 chars, 2 lines
    small = pd.DataFrame({"buggy": ["abc", "abcd", "a\nb", "a\nb\nc"], "fixed": ["a"] * 4})
    assert within_length_limit(small, max_chars=3, max_lines=2).tolist() == [True, False, True, False]
