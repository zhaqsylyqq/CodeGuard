"""Shared fixtures: a tiny synthetic CTSSB-like dataset written as gzipped JSONL shards."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

N_PROJECTS = 30
PAIRS_PER_PROJECT = 5


def make_diff(buggy: str, fixed: str, n_hunks: int = 1) -> str:
    hunk = (
        "@@ -10,7 +10,7 @@ def handler(request):\n"
        "     data = load(request)\n"
        "     if not data:\n"
        "         return None\n"
        f"-{buggy}\n"
        f"+{fixed}\n"
        "     log(result)\n"
        "     return result\n"
    )
    return hunk * n_hunks


def make_record(project: str, idx: int, buggy: str, fixed: str, **overrides) -> dict:
    record = {
        "project": project,
        "commit_sha": f"{project}-{idx:04d}",
        "parent_sha": "0" * 40,
        "file_path": f"{project}/module_{idx}.py",
        "project_url": f"https://github.com/owner/{project}",
        "likely_bug": True,
        "comodified": False,
        "in_function": True,
        "diff": make_diff(buggy, fixed),
        "before": buggy.strip(),
        "after": fixed.strip(),
        "sstub_pattern": ["CHANGE_IDENTIFIER_USED", "SINGLE_STMT", "CHANGE_BINARY_OPERAND"][
            idx % 3
        ],
        "edit_script": "[]",
    }
    record.update(overrides)
    return record


def synthetic_records() -> list[dict]:
    records = []
    for p in range(N_PROJECTS):
        project = f"proj{p:02d}"
        for i in range(PAIRS_PER_PROJECT):
            idx = p * PAIRS_PER_PROJECT + i
            records.append(
                make_record(
                    project,
                    idx,
                    buggy=f"    result = compute_{idx}(data, limit)",
                    fixed=f"    result = compute_{idx}(data, limit + 1)",
                    in_function=(idx % 7 != 0),  # some rows filtered out
                )
            )
    # Edge cases the pipeline must handle and count.
    records.append(
        make_record("proj00", 900, "    a = 1", "    a = 2", diff=make_diff("a", "b", 2))
    )
    records.append(make_record("proj01", 901, "    y = f(x)", "    y  =  f(x)"))  # whitespace only
    records.append(make_record("proj02", 902, "    print x", "    print(x)"))  # Python 2
    # A fork: different project name, identical change -> removed by deduplication.
    dup = dict(records[1], project="proj01_fork", project_url="https://github.com/fork/proj01")
    records.append(dup)
    return records


@pytest.fixture(scope="session")
def ctssb_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("ctssb")
    records = synthetic_records()
    half = len(records) // 2
    for k, chunk in enumerate([records[:half], records[half:]]):
        with gzip.open(root / f"file-{k}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r) + "\n")
    return root
