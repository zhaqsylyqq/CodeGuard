"""Task views derived on the fly from the stored pair table.

Storing *pairs* (one row per bug fix) and deriving per-task rows here
guarantees that the buggy and fixed halves of a pair always share a split.
"""

from __future__ import annotations

import pandas as pd

META_COLUMNS = ("pair_id", "project", "sstub_pattern", "is_py2")


def to_classification(pairs: pd.DataFrame) -> pd.DataFrame:
    """Two rows per pair: ``code=buggy, label=1`` and ``code=fixed, label=0``.

    Balanced by construction. Rows of the same pair are adjacent (buggy first).
    """
    meta = list(META_COLUMNS)
    buggy = pairs[meta].assign(code=pairs["buggy"].to_numpy(), label=1)
    fixed = pairs[meta].assign(code=pairs["fixed"].to_numpy(), label=0)
    rows = pd.concat([buggy, fixed], keys=[0, 1], names=["side", "row"])
    rows = rows.sort_index(level=["row", "side"]).reset_index(drop=True)
    return rows[[*meta, "code", "label"]]


def to_generation(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per pair: buggy snippet -> fixed snippet (and the fixed lines only)."""
    return pd.DataFrame(
        {
            **{c: pairs[c].to_numpy() for c in META_COLUMNS},
            "source": pairs["buggy"].to_numpy(),
            "target": pairs["fixed"].to_numpy(),
            "target_lines": pairs["added_lines"].to_numpy(),
        }
    )
