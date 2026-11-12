"""Shared data loading, run metadata, bootstrap intervals and classification reports."""

import json
import platform
import random
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from data_utils import pairs_to_examples

DEFAULT_DATA_DIR = "data/processed/small"
PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "sacrebleu", "pandas", "numpy")


def seed_everything(seed=42):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data


def load_pairs(data_dir, split, limit_pairs=None, seed=42):
    """Read one split of (buggy, fixed) pairs; optionally a seeded random subset of `limit_pairs`."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    path = Path(data_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run notebooks/01_data.ipynb first.")
    pairs = pd.read_parquet(path)
    if pairs.empty or not {"buggy", "fixed"}.issubset(pairs.columns):
        raise ValueError(f"{path} must contain nonempty buggy/fixed pairs")
    if pairs[["buggy", "fixed"]].isna().any().any():
        raise ValueError(f"Missing code in {path}")
    if limit_pairs is not None:
        if limit_pairs < 1:
            raise ValueError("limit_pairs must be positive")
        pairs = pairs.sample(min(limit_pairs, len(pairs)), random_state=seed)
    return pairs


def load_examples(data_dir, split, limit_pairs=None, seed=42):
    """Detection examples: every pair gives a buggy (1) and a fixed (0) example."""
    return pairs_to_examples(load_pairs(data_dir, split, limit_pairs, seed)).reset_index(drop=True)


# ---------------------------------------------------------------- uncertainty


def bootstrap_ci(values, n=2000, seed=42):
    """Mean and 95% percentile interval of the mean, resampling pairs with replacement."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("values must be nonempty")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high)}


def paired_bootstrap_diff(a, b, n=2000, seed=42):
    """Interval of mean(a - b) where a[i] and b[i] are two systems' scores on the same pair i."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired values must have the same shape")
    return bootstrap_ci(a - b, n=n, seed=seed)


# ---------------------------------------------------------------- run metadata


def package_versions():
    found = {}
    for package in PACKAGES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            pass
    return found


def git_commit():
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_names():
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def run_metadata(args):
    """Everything needed to say later exactly what produced a metrics.json."""
    return {
        "args": {key: value for key, value in vars(args).items()},
        "git_commit": git_commit(),
        "packages": package_versions(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpus": gpu_names(),
    }


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# ---------------------------------------------------------------- detection reports


def classification_metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or not len(labels):
        raise ValueError("Labels and probabilities must be nonempty and have equal shape")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid probabilities")
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "n_examples": len(labels),
    }


def save_report(output_dir, examples, probabilities, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {**metadata, "environment": package_versions(),
              **classification_metrics(examples.label, probabilities)}
    write_json(output_dir / "metrics.json", report)
    predictions = examples[["pair", "label"]].copy()
    predictions["p_buggy"] = probabilities
    predictions["prediction"] = (np.asarray(probabilities) >= 0.5).astype(int)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    return report
