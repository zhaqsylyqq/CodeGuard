"""The shared results table (results/results.csv) that every experiment appends to."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

RESULT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "run_id",
    "tier",
    "task",
    "model",
    "dataset",
    "split",
    "n",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "pair_accuracy",
    "exact_match",
    "bleu",
    "codebleu",
    "train_seconds",
    "git_hash",
    "notes",
)


def append_result(row: dict[str, Any], path: Path) -> None:
    """Append one row; unknown columns are an error so the schema stays fixed."""
    unknown = set(row) - set(RESULT_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown result columns: {sorted(unknown)}")
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
    row = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, restval="")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
