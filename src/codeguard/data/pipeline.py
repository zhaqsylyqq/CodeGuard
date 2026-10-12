"""End-to-end data preparation: raw records -> leakage-safe train/val/test pairs.

    read -> filter (likely_bug, in_function) -> rebuild snippets   [cached "pool"]
         -> single hunk -> real change -> (optional) Python 3 only
         -> deduplicate -> grouped split -> drop cross-split duplicates
         -> nested sample -> {train,val,test}.parquet + manifest.json

The expensive first part (reading ~1M JSON records) is cached as a parquet
"pool" keyed by the source and the record-level filters, so preparing a 5k and
a 50k dataset reads the raw shards only once.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

import codeguard.data.sources  # noqa: F401  (registers built-in sources)
from codeguard.config import DataConfig, dump_config
from codeguard.data.diff import reconstruct
from codeguard.data.quality import code_hash, python2_markers
from codeguard.data.split import (
    SPLITS,
    assert_no_leakage,
    assign_splits,
    cross_split_duplicates,
    drop_cross_split_duplicates,
    leakage_report,
    sample_pairs,
)
from codeguard.registry import DATASETS
from codeguard.utils.run import collect_env, get_logger, git_info, setup_logging

# Bump when pool construction changes, to invalidate stale caches.
POOL_VERSION = "1"
CHUNK_ROWS = 100_000

OUTPUT_COLUMNS = [
    "pair_id",
    "project",
    "project_url",
    "file_path",
    "split",
    "sstub_pattern",
    "buggy",
    "fixed",
    "removed_lines",
    "added_lines",
    "before",
    "after",
    "n_lines",
    "is_py2",
    "py2_markers",
    "buggy_hash",
    "fixed_hash",
]


# ------------------------------------------------------------------------- pool


def _pool_cache_paths(cfg: DataConfig) -> tuple[Path, Path, dict[str, Any]]:
    key = {
        "pool_version": POOL_VERSION,
        "source": cfg.source.name,
        "path": str(cfg.source.path.resolve()),
        "max_rows": cfg.source.max_rows,
        "likely_bug": cfg.filters.likely_bug,
        "in_function": cfg.filters.in_function,
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    stem = cfg.cache_dir / f"{cfg.source.name}_pool_{digest}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".json"), key


def _pool_row(rec: dict[str, Any]) -> dict[str, Any]:
    pair = reconstruct(rec["diff"])
    markers = python2_markers(pair.buggy, pair.fixed)
    pair_id = hashlib.blake2b(
        f"{rec.get('commit_sha')}\x00{rec.get('file_path')}\x00{rec['diff']}".encode(),
        digest_size=8,
    ).hexdigest()
    return {
        "pair_id": pair_id,
        "project": rec["project"],
        "project_url": rec.get("project_url"),
        "file_path": rec.get("file_path"),
        "sstub_pattern": rec.get("sstub_pattern"),
        "buggy": pair.buggy,
        "fixed": pair.fixed,
        "removed_lines": "\n".join(pair.removed),
        "added_lines": "\n".join(pair.added),
        "before": rec.get("before"),
        "after": rec.get("after"),
        "n_hunks": pair.n_hunks,
        "n_lines": pair.buggy.count("\n") + 1,
        "is_py2": bool(markers),
        "py2_markers": ",".join(markers),
        "buggy_hash": code_hash(pair.buggy),
        "fixed_hash": code_hash(pair.fixed),
    }


def build_pool(cfg: DataConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    """Stream raw records, apply record-level filters and rebuild snippets."""
    reader = DATASETS.get(cfg.source.name)
    counts: Counter[str] = Counter()
    chunks: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for rec in tqdm(reader(cfg.source), desc="reading records", unit="rec", mininterval=5):
        counts["raw"] += 1
        if cfg.filters.likely_bug is not None and rec.get("likely_bug") != cfg.filters.likely_bug:
            continue
        counts["likely_bug"] += 1
        if (
            cfg.filters.in_function is not None
            and rec.get("in_function") != cfg.filters.in_function
        ):
            continue
        counts["in_function"] += 1
        rows.append(_pool_row(rec))
        if len(rows) >= CHUNK_ROWS:
            chunks.append(pd.DataFrame(rows))
            rows = []
    if rows:
        chunks.append(pd.DataFrame(rows))
    if not chunks:
        raise ValueError("No records left after record-level filters")
    return pd.concat(chunks, ignore_index=True), dict(counts)


def load_or_build_pool(cfg: DataConfig) -> tuple[pd.DataFrame, dict[str, int], Path]:
    """Return the cached pool if its key matches, otherwise build and cache it."""
    log = get_logger()
    pool_path, meta_path, key = _pool_cache_paths(cfg)
    if pool_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            log.info("Using cached pool %s", pool_path)
            return pd.read_parquet(pool_path), meta["counts"], pool_path

    log.info("Building pool from %s (this reads every raw record once)", cfg.source.path)
    pool, counts = build_pool(cfg)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(pool_path, index=False)
    meta_path.write_text(json.dumps({"key": key, "counts": counts}, indent=2), encoding="utf-8")
    log.info("Cached pool (%d pairs) -> %s", len(pool), pool_path)
    return pool, counts, pool_path


# ---------------------------------------------------------------------- prepare


def content_hash(df: pd.DataFrame) -> str:
    """Hash of the table contents (independent of parquet encoding details)."""
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=False).to_numpy()).hexdigest()[:16]


def _split_stats(df: pd.DataFrame, group_key: str) -> dict[str, dict[str, Any]]:
    stats = {}
    for split in SPLITS:
        part = df[df["split"] == split]
        stats[split] = {
            "pairs": len(part),
            "groups": int(part[group_key].nunique()),
            "row_fraction": round(len(part) / max(len(df), 1), 4),
            "python2_share": round(float(part["is_py2"].mean()), 4) if len(part) else None,
        }
    return stats


def _marker_counts(df: pd.DataFrame) -> dict[str, int]:
    exploded = df["py2_markers"].str.split(",").explode()
    return {k: int(v) for k, v in exploded[exploded != ""].value_counts().items()}


def prepare(cfg: DataConfig) -> dict[str, Any]:
    """Run the full data pipeline for ``cfg`` and return the manifest."""
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "prepare.log")
    t0 = time.perf_counter()
    gk = cfg.split.group_key

    pool, counts, pool_path = load_or_build_pool(cfg)
    funnel: dict[str, int] = {
        "raw_records": counts["raw"],
        "likely_bug": counts["likely_bug"],
        "in_function": counts["in_function"],
    }

    df = pool
    if cfg.filters.single_hunk:
        df = df[df["n_hunks"] == 1]
        funnel["single_hunk"] = len(df)
    df = df[df["buggy_hash"] != df["fixed_hash"]]  # drops whitespace-only "fixes"
    funnel["non_whitespace_change"] = len(df)
    if cfg.filters.drop_python2:
        df = df[~df["is_py2"]]
        funnel["python3_only"] = len(df)
    df = df.drop_duplicates(["buggy_hash", "fixed_hash"], keep="first")
    funnel["deduplicated"] = len(df)

    df = df.reset_index(drop=True)
    df["split"] = assign_splits(df[gk], cfg.split.val, cfg.split.test, cfg.seed)
    duplicates_before = cross_split_duplicates(df)
    dropped = {"val": 0, "test": 0}
    if cfg.split.drop_cross_split_duplicates:
        df, dropped = drop_cross_split_duplicates(df)
        funnel["no_cross_split_duplicates"] = len(df)
    pool_leakage = leakage_report(df, gk)
    assert_no_leakage(pool_leakage)

    fractions = {"val": cfg.split.val, "test": cfg.split.test}
    sample, warnings = sample_pairs(
        df, cfg.sample.n_pairs, fractions, cfg.seed, cfg.sample.max_pairs_per_project, gk
    )
    funnel["sampled"] = len(sample)
    for w in warnings:
        log.warning(w)
    sample_leakage = leakage_report(sample, gk)
    assert_no_leakage(sample_leakage)

    files = {}
    for split in SPLITS:
        part = sample.loc[sample["split"] == split, OUTPUT_COLUMNS].reset_index(drop=True)
        path = out / f"{split}.parquet"
        part.to_parquet(path, index=False)
        files[split] = {
            "path": path.as_posix(),
            "rows": len(part),
            "content_hash": content_hash(part),
        }

    urls_per_project = df.groupby(gk)["project_url"].nunique()
    manifest = {
        "name": cfg.name,
        "tier": cfg.tier,
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "duration_seconds": round(time.perf_counter() - t0, 1),
        "pool_cache": pool_path.as_posix(),
        "funnel": funnel,
        "files": files,
        "pool": {
            "pairs": len(df),
            "groups": int(df[gk].nunique()),
            "groups_with_multiple_urls (forks)": int((urls_per_project > 1).sum()),
            "python2_share": round(float(df["is_py2"].mean()), 4),
            "python2_markers": _marker_counts(df),
            "splits": _split_stats(df, gk),
        },
        "sample": {
            "python2_share": round(float(sample["is_py2"].mean()), 4),
            "python2_markers": _marker_counts(sample),
            "splits": _split_stats(sample, gk),
            "warnings": warnings,
        },
        "leakage": {
            "cross_split_duplicates_before_drop": duplicates_before,
            "dropped_cross_split_duplicates": dropped,
            "pool": pool_leakage,
            "sample": sample_leakage,
        },
        "git": git_info(),
        "env": collect_env(),
    }
    dump_config(cfg, out / "config.yaml")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Prepared %s: %s", cfg.name, {s: f["rows"] for s, f in files.items()})
    return manifest


def load_pairs(data_dir: Path, split: str) -> pd.DataFrame:
    """Load one split of a prepared dataset."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    return pd.read_parquet(data_dir / f"{split}.parquet")
