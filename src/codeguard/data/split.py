"""Leakage-safe splitting and sampling of bug-fix pairs.

Order of operations (see docs/data.md for the reasoning):

1. ``assign_splits``: GroupShuffleSplit over the *whole* filtered pool, so a
   project (all its forks share the name) lives in exactly one split.
2. ``drop_cross_split_duplicates``: remove val/test pairs whose buggy or fixed
   snippet also occurs in an earlier split (train < val < test).
3. ``sample_pairs``: a fixed per-split permutation, then take a prefix. Samples
   of different sizes are therefore nested (the 5k sample is inside the 50k one).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SPLITS = ("train", "val", "test")


def assign_splits(groups: pd.Series, val: float, test: float, seed: int) -> pd.Series:
    """Return a train/val/test label per row; whole groups go to one split.

    ``val`` and ``test`` are fractions of *groups*, not rows, so row fractions
    are approximate (large projects make them drift). Report the actual ones.
    """
    groups = groups.reset_index(drop=True)
    index = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    trainval_idx, test_idx = next(outer.split(index, groups=groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=val / (1 - test), random_state=seed)
    train_rel, val_rel = next(inner.split(trainval_idx, groups=groups.iloc[trainval_idx]))

    labels = np.empty(len(groups), dtype=object)
    labels[test_idx] = "test"
    labels[trainval_idx[train_rel]] = "train"
    labels[trainval_idx[val_rel]] = "val"
    return pd.Series(labels, name="split")


def _snippet_hashes(df: pd.DataFrame) -> set[str]:
    return set(df["buggy_hash"].to_numpy(dtype=object)) | set(
        df["fixed_hash"].to_numpy(dtype=object)
    )


def _overlaps(df: pd.DataFrame, seen: set[str]) -> np.ndarray:
    """Rows whose buggy or fixed hash is in ``seen``.

    Plain set lookups: ``Series.isin(set)`` on Arrow-backed strings (pandas 3)
    re-converts the whole set on every call and is ~100x slower here.
    """
    buggy = df["buggy_hash"].to_numpy(dtype=object)
    fixed = df["fixed_hash"].to_numpy(dtype=object)
    return np.fromiter(
        (b in seen or f in seen for b, f in zip(buggy, fixed, strict=True)), bool, len(df)
    )


def cross_split_duplicates(df: pd.DataFrame) -> dict[str, int]:
    """Pairs in the later split that share a buggy or fixed snippet with the earlier one."""
    out = {}
    for a, b in combinations(SPLITS, 2):
        seen = _snippet_hashes(df[df["split"] == a])
        later = df[df["split"] == b]
        out[f"{a}-{b}"] = int(_overlaps(later, seen).sum())
    return out


def drop_cross_split_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop val pairs that overlap train, then test pairs that overlap train or val."""
    keep = pd.Series(True, index=df.index)
    dropped = {}
    seen = _snippet_hashes(df[df["split"] == "train"])
    for split in ("val", "test"):
        part = df[df["split"] == split]
        hit = _overlaps(part, seen)
        keep[part.index[hit]] = False
        dropped[split] = int(hit.sum())
        seen |= _snippet_hashes(part[~hit])
    return df[keep], dropped


def sample_pairs(
    df: pd.DataFrame,
    n_pairs: int | None,
    fractions: dict[str, float],
    seed: int,
    max_per_group: int | None = None,
    group_key: str = "project",
) -> tuple[pd.DataFrame, list[str]]:
    """Take a nested, per-split sample with ``fractions`` of ``n_pairs`` per split.

    Each split is shuffled with its own seeded generator and a prefix is taken,
    so for the same pool a smaller ``n_pairs`` always yields a subset of a larger one.
    Returns the sample and a list of warnings (e.g. a split smaller than requested).
    """
    warnings: list[str] = []
    targets = _split_targets(n_pairs, fractions)
    parts = []
    for k, split in enumerate(SPLITS):
        part = df[df["split"] == split]
        rng = np.random.default_rng([seed, k])
        part = part.iloc[rng.permutation(len(part))]
        if max_per_group is not None:
            part = part[part.groupby(group_key).cumcount() < max_per_group]
        target = targets.get(split)
        if target is not None:
            if target > len(part):
                warnings.append(f"{split}: requested {target} pairs, only {len(part)} available")
            part = part.head(target)
        parts.append(part)
    return pd.concat(parts, ignore_index=True), warnings


def _split_targets(n_pairs: int | None, fractions: dict[str, float]) -> dict[str, int]:
    if n_pairs is None:
        return {}
    val = round(n_pairs * fractions["val"])
    test = round(n_pairs * fractions["test"])
    return {"train": n_pairs - val - test, "val": val, "test": test}


def leakage_report(df: pd.DataFrame, group_key: str = "project") -> dict[str, Any]:
    """Group overlap and snippet-level duplicates between splits of ``df``."""
    groups = {s: set(df.loc[df["split"] == s, group_key]) for s in SPLITS}
    pairs_per_id = df.groupby("pair_id")["split"].nunique()
    buggy = set(df["buggy_hash"].to_numpy(dtype=object))
    fixed = set(df["fixed_hash"].to_numpy(dtype=object))
    return {
        "shared_groups": {
            f"{a}-{b}": len(groups[a] & groups[b]) for a, b in combinations(SPLITS, 2)
        },
        "pairs_in_multiple_splits": int((pairs_per_id > 1).sum()),
        "cross_split_snippet_duplicates": cross_split_duplicates(df),
        # same text is "buggy" in one pair and "fixed" in another (e.g. reverts):
        # irreducible label noise for the classification task
        "label_conflict_snippets": len(buggy & fixed),
    }


def assert_no_leakage(report: dict[str, Any]) -> None:
    """Fail loudly if a project or a pair crosses splits."""
    shared = {k: v for k, v in report["shared_groups"].items() if v}
    if shared:
        raise AssertionError(f"Groups shared between splits: {shared}")
    if report["pairs_in_multiple_splits"]:
        raise AssertionError(f"{report['pairs_in_multiple_splits']} pairs cross splits")
