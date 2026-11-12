"""Reusable helpers for preparing the CTSSB bug-fix dataset.

Pipeline (see notebooks/01_data.ipynb):
    load shards -> rebuild buggy/fixed code from diff -> dedup -> split by project
    -> remove cross-split duplicates -> length filter -> nested samples (small, full).
"""

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

# We only need these columns; the rest (edit_script, commit ids, ...) is big and unused.
KEEP_COLUMNS = ["project", "file_path", "likely_bug", "in_function", "diff", "sstub_pattern"]

SPLITS = ["train", "val", "test"]

# Simple regex markers of Python 2 code (valid in Python 2, not in Python 3).
PY2_PATTERNS = [
    re.compile(r"^\s*print\s+[^\s(=,.)\[]", re.MULTILINE),  # print "text"
    re.compile(r"\bexcept\s+[\w.]+\s*,\s*\w+\s*:"),  # except Error, e:
    re.compile(r"\braise\s+[\w.]+\s*,"),  # raise Error, "msg"
    re.compile(r"\b\d+[lL]\b"),  # long literal: 23L
    re.compile(r"\.has_key\("),
    re.compile(r"\bxrange\("),
    re.compile(r"\.(?:iteritems|iterkeys|itervalues)\("),
    re.compile(r"\bbasestring\b"),
]


# ---------------------------------------------------------------- loading


def load_pool(raw_dir, pool_path):
    """Read all shards, keep needed columns and in_function rows; cache as parquet.

    Reading ~1M JSON records takes a while, so we do it once and reuse the file.
    Returns (pool, n_raw) where n_raw is the number of records before filtering.
    """
    pool_path = Path(pool_path)
    info_path = pool_path.with_suffix(".json")
    if pool_path.exists() and info_path.exists():
        n_raw = json.loads(info_path.read_text())["n_raw"]
        return pd.read_parquet(pool_path), n_raw

    raw_dir = Path(raw_dir)
    shards_by_number = {}
    for pattern in ("file-*.jsonl", "file-*.jsonl.gz"):
        for path in raw_dir.glob(pattern):
            match = re.fullmatch(r"file-(\d+)\.jsonl(?:\.gz)?", path.name)
            if match:
                shards_by_number[int(match[1])] = path
    shards = [shards_by_number[number] for number in sorted(shards_by_number)]
    if not shards:
        raise FileNotFoundError(
            f"No file-*.jsonl or file-*.jsonl.gz shards found in {raw_dir.resolve()}. "
            "Place the CTSSB files there or update RAW_DIR in the notebook."
        )
    parts, n_raw = [], 0
    for shard in tqdm(shards, desc="reading shards"):
        open_shard = gzip.open if shard.suffix == ".gz" else open
        with open_shard(shard, "rt", encoding="utf-8") as f:
            df = pd.read_json(f, lines=True, dtype=False)[
                KEEP_COLUMNS
            ]  # dtype=False: keep text as text
        n_raw += len(df)
        parts.append(df[df["in_function"]])  # filter per shard to save memory

    pool = pd.concat(parts, ignore_index=True)
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(pool_path, index=False)
    info_path.write_text(json.dumps({"n_raw": n_raw}))
    return pool, n_raw


# ------------------------------------------------------- rebuild code from diff


def rebuild_snippets(diff):
    """Turn a unified diff into (buggy, fixed, n_hunks).

    ' ' lines go to both versions, '-' only to buggy, '+' only to fixed.
    The first character is dropped. '@@' headers and '\\ No newline' are skipped.
    """
    lines = diff.replace("\r\n", "\n").split("\n")
    if lines[-1] == "":
        lines.pop()  # the diff ends with a newline, not with a real empty line

    buggy, fixed, n_hunks = [], [], 0
    for line in lines:
        if line.startswith("@@"):
            n_hunks += 1
        elif line.startswith("\\"):
            continue
        elif line.startswith("-"):
            buggy.append(line[1:])
        elif line.startswith("+"):
            fixed.append(line[1:])
        else:  # context line (an empty line may have lost its leading space)
            buggy.append(line[1:])
            fixed.append(line[1:])
    return "\n".join(buggy), "\n".join(fixed), n_hunks


def add_snippets(df):
    """Add buggy, fixed and n_hunks columns to a table that has a diff column."""
    rebuilt = [rebuild_snippets(d) for d in tqdm(df["diff"], desc="rebuilding code")]
    df = df.copy()
    df["buggy"] = [r[0] for r in rebuilt]
    df["fixed"] = [r[1] for r in rebuilt]
    df["n_hunks"] = [r[2] for r in rebuilt]
    return df


# ------------------------------------------------------------------ hashing


def normalize_code(code):
    """Strip trailing spaces and drop blank lines, but KEEP leading indentation.

    Indentation matters in Python: a fix that only changes indentation is a real
    fix, so it must not look identical to the buggy code.
    """
    lines = [line.rstrip() for line in code.split("\n")]
    return "\n".join(line for line in lines if line)


def code_hash(code):
    """Short fingerprint of normalized code, used to find duplicates."""
    return hashlib.md5(normalize_code(code).encode("utf-8")).hexdigest()


def add_hashes(df):
    """Add buggy_hash and fixed_hash columns."""
    df = df.copy()
    df["buggy_hash"] = [code_hash(c) for c in df["buggy"]]
    df["fixed_hash"] = [code_hash(c) for c in df["fixed"]]
    return df


# ------------------------------------------------------------------ python 2


def is_python2(code):
    """True if the code contains at least one Python 2 marker (a heuristic)."""
    return any(p.search(code) for p in PY2_PATTERNS)


def add_python2_flag(df):
    """Mark pairs whose buggy or fixed code looks like Python 2 (we do not drop them)."""
    df = df.copy()
    df["is_py2"] = df["buggy"].map(is_python2) | df["fixed"].map(is_python2)
    return df


# -------------------------------------------------------------------- split


def split_by_project(df, seed=42, val=0.1, test=0.1):
    """Label each row train/val/test so that one project lives in one split only.

    Why by project: code from one project is very similar, so a random row split
    would leak near-copies from train into test and inflate the scores.
    Two steps: cut off test first, then cut val from what is left.
    """
    projects = df["project"]
    outer = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    trainval_idx, test_idx = next(outer.split(df, groups=projects))

    rest = projects.iloc[trainval_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=val / (1 - test), random_state=seed)
    train_pos, val_pos = next(inner.split(rest, groups=rest))

    labels = np.empty(len(df), dtype=object)
    labels[test_idx] = "test"
    labels[trainval_idx[train_pos]] = "train"
    labels[trainval_idx[val_pos]] = "val"
    return pd.Series(labels, index=df.index, name="split")


def _hashes(df):
    return set(df["buggy_hash"]) | set(df["fixed_hash"])


def _overlaps(df, seen):
    """Boolean array: row's buggy or fixed hash is in the set `seen`."""
    buggy_hit = np.array([h in seen for h in df["buggy_hash"]], dtype=bool)
    fixed_hit = np.array([h in seen for h in df["fixed_hash"]], dtype=bool)
    return buggy_hit | fixed_hit


def remove_cross_split_duplicates(df):
    """Drop val rows seen in train, then test rows seen in train or val.

    Different projects can contain the same code (copy-paste, vendored libraries).
    Such rows would let the model "remember" the test answer from train.
    Returns (clean_df, {"val": n_dropped, "test": n_dropped}).
    """
    seen = _hashes(df[df["split"] == "train"])
    keep = pd.Series(True, index=df.index)
    dropped = {}
    for name in ["val", "test"]:
        part = df[df["split"] == name]
        hit = _overlaps(part, seen)
        keep[part.index[hit]] = False
        dropped[name] = int(hit.sum())
        seen |= _hashes(part[~hit])
    return df[keep], dropped


def check_leakage(df):
    """Count projects and snippets shared between splits. Raises if a project is shared."""
    report = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            part_a, part_b = df[df["split"] == a], df[df["split"] == b]
            shared_projects = set(part_a["project"]) & set(part_b["project"])
            report[f"{a}-{b}"] = {
                "shared_projects": len(shared_projects),
                "shared_snippets": int(_overlaps(part_b, _hashes(part_a)).sum()),
            }
    assert all(r["shared_projects"] == 0 for r in report.values()), "a project is in two splits"
    return report


# ----------------------------------------------------------------- sampling


def sample_pairs(df, sizes, seed=42):
    """Random sample per split, e.g. sizes={"train": 4000, "val": 500, "test": 500}."""
    parts = [df[df["split"] == name].sample(n=n, random_state=seed) for name, n in sizes.items()]
    return pd.concat(parts, ignore_index=True)


def sample_nested(df, sizes_by_tier, seed=42):
    """Nested samples per split, e.g.
    sizes_by_tier={"small": {"train": 4000, ...}, "full": {"train": 50000, ...}}.

    Each split is shuffled ONCE with the seed and every tier takes a prefix of
    that order. So the smaller tier is a strict subset of the bigger one within
    every split: results on "small" are directly comparable to "full" (same
    pairs, just fewer), and nothing in "small" is unseen by "full".
    Returns {tier_name: DataFrame} with a fresh index and the original columns.
    """
    shuffled = {}
    for sizes in sizes_by_tier.values():
        for name, n in sizes.items():
            if name not in shuffled:
                part = df[df["split"] == name]
                shuffled[name] = part.sample(frac=1, random_state=seed)
            if n > len(shuffled[name]):
                raise ValueError(f"split {name!r} has {len(shuffled[name])} rows, {n} requested")
    return {
        tier: pd.concat([shuffled[name].iloc[:n] for name, n in sizes.items()], ignore_index=True)
        for tier, sizes in sizes_by_tier.items()
    }


# ------------------------------------------------------------ length filter

MAX_SNIPPET_CHARS = 2000
MAX_SNIPPET_LINES = 60


def _within(code, max_chars, max_lines):
    return len(code) <= max_chars and code.count("\n") + 1 <= max_lines


def within_length_limit(df, max_chars=MAX_SNIPPET_CHARS, max_lines=MAX_SNIPPET_LINES):
    """Boolean Series: True when BOTH buggy and fixed are within both limits.

    Counted in characters and lines, not model tokens, so the filter does not
    depend on which tokenizer is used later. Removes rare very long outliers.
    """
    buggy_ok = df["buggy"].map(lambda code: _within(code, max_chars, max_lines))
    fixed_ok = df["fixed"].map(lambda code: _within(code, max_chars, max_lines))
    return (buggy_ok & fixed_ok).astype(bool)


# ------------------------------------------------------------ classification


def pairs_to_examples(df):
    """Turn each (buggy, fixed) pair into two labelled examples: buggy=1, fixed=0.

    Each pair gives one positive and one negative example, so the classes are
    balanced by construction (50/50). `pair` links both examples of one pair.
    """
    buggy = pd.DataFrame({"pair": df.index, "text": df["buggy"].to_numpy(), "label": 1})
    fixed = pd.DataFrame({"pair": df.index, "text": df["fixed"].to_numpy(), "label": 0})
    return pd.concat([buggy, fixed], ignore_index=True)
