import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_utils import (
    check_leakage,
    code_hash,
    is_python2,
    normalize_code,
    rebuild_snippets,
    remove_cross_split_duplicates,
    split_by_project,
)

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
