"""Shared input checks and classification reports for neural experiments."""

import json
import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from data_utils import pairs_to_examples


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(data_dir, split, limit_pairs=None, seed=42):
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
    return pairs_to_examples(pairs).reset_index(drop=True)


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
    environment = {}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            pass
    report = {**metadata, "environment": environment, **classification_metrics(examples.label, probabilities)}
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    predictions = examples[["pair", "label"]].copy()
    predictions["p_buggy"] = probabilities
    predictions["prediction"] = (np.asarray(probabilities) >= 0.5).astype(int)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    return report
