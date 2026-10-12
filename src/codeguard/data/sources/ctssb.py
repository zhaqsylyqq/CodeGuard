"""Reader for CTSSB-1M / TSSB-3M shards (``file-*.jsonl.gz``).

A source is any callable ``(SourceConfig) -> Iterator[dict]`` registered in
``DATASETS`` that yields raw records with at least ``project``, ``diff`` and
``sstub_pattern``. The pipeline takes care of everything else.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from codeguard.config import SourceConfig
from codeguard.registry import DATASETS

KEEP_FIELDS = (
    "project",
    "project_url",
    "commit_sha",
    "file_path",
    "likely_bug",
    "comodified",
    "in_function",
    "diff",
    "before",
    "after",
    "sstub_pattern",
)  # `edit_script` and `parent_sha` are dropped: large and unused


def shard_paths(root: Path) -> list[Path]:
    """Shards sorted numerically (file-2 before file-10), so reading order is stable."""
    paths = list(root.glob("file-*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"No file-*.jsonl.gz shards found in {root}")
    return sorted(paths, key=lambda p: int(re.search(r"(\d+)", p.name).group(1)))


@DATASETS.register("ctssb")
def read_ctssb(cfg: SourceConfig) -> Iterator[dict[str, Any]]:
    """Stream records from all shards, keeping only ``KEEP_FIELDS``."""
    remaining = cfg.max_rows
    for path in shard_paths(cfg.path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if remaining is not None:
                    if remaining == 0:
                        return
                    remaining -= 1
                record = json.loads(line)
                yield {k: record.get(k) for k in KEEP_FIELDS}
