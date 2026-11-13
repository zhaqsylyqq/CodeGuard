"""FAKE run directories for tests/test_report.py and for checking notebooks/06_results.ipynb.

Every number written here is invented to exercise the report code. These are NOT results.
Write them only to a temporary directory (pytest tmp_path), never to runs/ or results/.
Every metrics.json carries "fixture": FAKE so a stray copy is recognisable.
"""

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

FAKE = "FAKE TEST FIXTURE - invented numbers, not a result"
M13, M67, M33 = ("fixture/coder-1.3b-instruct", "fixture/coder-6.7b-instruct", "fixture/coder-33b-instruct")

# name: (method, model, tier, rank, val exact-match rate, test exact-match rate)
# lora_6.7b_r16 is best on val among full-tier non-replicate LoRA runs; lora_6.7b_r8 is best on test.
REPAIR = {
    "zs_1.3b": ("zero-shot", M13, "full", None, 0.05, 0.05),
    "zs_6.7b": ("zero-shot", M67, "full", None, 0.10, 0.10),
    "fs3_6.7b": ("3-shot", M67, "full", None, 0.15, 0.15),
    "lora_6.7b_r16": ("lora", M67, "full", 16, 0.30, 0.20),
    "lora_6.7b_r8": ("lora", M67, "full", 8, 0.25, 0.45),
    "qlora_6.7b_r16": ("qlora", M67, "full", 16, 0.275, 0.25),
    "lora_1.3b_r16": ("lora", M13, "full", 16, 0.225, 0.20),
    "lora_6.7b_r16_s43": ("lora", M67, "full", 16, 0.50, 0.50),
    "zs_33b": ("zero-shot", M33, "full", None, 0.20, 0.20),
    "zs_1.3b_small": ("zero-shot", M13, "small", None, 0.05, 0.05),
    "lora_1.3b_r16_small": ("lora", M13, "small", 16, 0.60, 0.60),
}
PATTERNS = ["CHANGE_IDENTIFIER_USED"] * 40 + ["SWAP_BOOLEAN_LITERAL"] * 12 + ["CHANGE_NUMERIC_LITERAL"] * 8


def pairs(split, tier):
    """Pair ids, bug types and code; the small tier is a prefix of the full tier (nested)."""
    n = {"val": 40, "test": 60}[split] // (2 if tier == "small" else 1)
    start = {"val": 1000, "test": 2000}[split]
    rows = []
    for i in range(n):
        pid = start + i
        rows.append({"pair_id": pid, "sstub_pattern": PATTERNS[(i * 7) % 60], "is_py2": i % 10 == 0,
                     "buggy": f"def f{pid}(x):\n    y = x - 1\n    return y",
                     "fixed": f"def f{pid}(x):\n    y = x + 1\n    return y"})
    return pd.DataFrame(rows)


def predictions(name, split, tier, rate, n=None):
    table = pairs(split, tier)
    if n is not None:
        table = table.head(n)
    rng = np.random.default_rng(zlib.crc32(f"{name}/{split}".encode()))
    k = len(table)
    order = rng.permutation(k)
    exact = np.zeros(k, bool)
    exact[order[:round(rate * k)]] = True
    localized = exact | (rng.random(k) < 0.3)
    changed = localized | (rng.random(k) < 0.5)
    wrong = "    y = x * 1"
    table["prediction"] = [f if e else (b.replace("    y = x - 1", wrong) if c else b)
                           for b, f, e, c in zip(table["buggy"], table["fixed"], exact, changed)]
    table["raw_output"] = "```python\n" + table["prediction"] + "\n```"
    table["parse_ok"] = True
    table["n_new_tokens"] = 20
    table["exact_match"], table["changed"] = exact, changed
    table["localized"], table["localized_exact"] = localized, localized
    table["s_fwd"] = rng.normal(-10, 1, k)
    table["s_bwd"] = rng.normal(-10.3, 1, k)
    table["direction_correct"] = table["s_fwd"] > table["s_bwd"]
    return table


def rates(frame):
    out = {"n": len(frame)}
    for column in ("exact_match", "changed", "localized", "localized_exact", "parse_ok", "direction_correct"):
        out[column] = float(frame[column].astype(float).mean())
    return out


def summary(table):
    from experiment_utils import bootstrap_ci

    overall = {**rates(table), "bleu": 50.0, "copy_bleu": 60.0, "copy_exact_match": 0.0,
               "localization_chance": 0.33, "direction_chance": 0.5}
    counts = table["sstub_pattern"].value_counts()
    pattern = table["sstub_pattern"].where(table["sstub_pattern"].map(counts) >= 30, "OTHER")
    return {
        "overall": overall,
        "intervals": {c: bootstrap_ci(table[c].astype(float)) for c in ("exact_match", "localized",
                                                                       "direction_correct")},
        "by_pattern": {p: rates(g) for p, g in table.groupby(pattern)},
        "by_py2": {str(bool(k)).lower(): rates(g) for k, g in table.groupby(table["is_py2"].astype(bool))},
        "eval_seconds": 10.0, "pairs_per_second": len(table) / 10.0,
    }


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def log_history(steps=40):
    rows = []
    for step in range(10, steps + 1, 10):
        rows.append({"loss": 1.0 / step ** 0.3, "step": step, "epoch": step / steps * 2})
        if step % 20 == 0:
            rows.append({"eval_loss": 1.1 / step ** 0.25, "step": step, "epoch": step / steps * 2})
    return rows


def make_runs(root, test_rates=None):
    """Write fake runs under root/runs and a fake TF-IDF file under root/results; returns (runs, results).

    `test_rates` overrides test exact-match rates per run (to check that selection ignores test).
    """
    root = Path(root)
    runs, results = root / "runs", root / "results"
    test_rates = test_rates or {}
    for name, (method, model, tier, rank, val_rate, test_rate) in REPAIR.items():
        out = runs / name
        splits = {}
        for split, rate in (("val", val_rate), ("test", test_rates.get(name, test_rate))):
            n = 50 if (name == "zs_6.7b" and split == "test") else None  # partial overlap on test
            table = predictions(name, split, tier, rate, n)
            out.mkdir(parents=True, exist_ok=True)
            table.to_parquet(out / f"{split}_predictions.parquet", index=False)
            splits[split] = summary(table)
        metrics = {"fixture": FAKE, "run": name, "task": "repair", "method": method, "model": model,
                   "revision": None, "quantized": method == "qlora", "shots": 3 if method == "3-shot" else 0,
                   "splits": splits, "eval_seconds": 20.0, "total_parameters": 6_700_000_000,
                   "trainable_parameters": 6_700_000_000,
                   "args": {"data_dir": f"data/processed/{tier}"}}
        if method in ("lora", "qlora"):
            metrics.update({"rank": rank, "adapter": str(out / "adapter"), "train_pairs": 100,
                            "train_examples": 98, "train_seconds": 600.0, "train_samples_per_second": 3.3,
                            "trainable_parameters": rank * 2_500_000, "adapter_bytes": rank * 10_000_000,
                            "log_history": log_history(), "global_step": 40, "epochs_done": 2.0,
                            "stopped_by_time_budget": False, "best_eval_loss": 0.4})
        write(out / "metrics.json", metrics)

    # A run with fields missing: no rank, no test split, no intervals, no cost fields.
    val = predictions("lora_6.7b_r32", "val", "full", 0.2)
    write(runs / "lora_6.7b_r32" / "metrics.json",
          {"fixture": FAKE, "task": "repair", "method": "lora", "model": M67,
           "args": {"data_dir": "data/processed/full"},
           "splits": {"val": {"overall": rates(val)}}})
    # Legacy detection runs (one without a "task" field) and an unfinished run without metrics.json.
    write(runs / "cls_zs_6.7b" / "metrics.json",
          {"fixture": FAKE, "model": M67, "adapter": None, "split": "val", "accuracy": 0.51, "precision": 0.5,
           "recall": 0.6, "f1": 0.55, "roc_auc": 0.52, "n_examples": 80, "args": {"data_dir": "data/processed/full"}})
    write(runs / "cls_lora_6.7b_r16" / "metrics.json",
          {"fixture": FAKE, "task": "detection", "model": M67, "method": "lora", "rank": 16, "split": "val",
           "accuracy": 0.55, "precision": 0.56, "recall": 0.5, "f1": 0.53, "roc_auc": 0.57, "n_examples": 80})
    write(runs / "bilstm" / "metrics.json",
          {"fixture": FAKE, "task": "detection", "model": "bilstm", "split": "val", "accuracy": 0.5,
           "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5, "n_examples": 80})
    (runs / "unfinished_run").mkdir(parents=True, exist_ok=True)
    (runs / "unfinished_run" / "train.log").write_text("crashed\n", encoding="utf-8")
    write(results / "baseline_metrics.json",
          {"fixture": FAKE, "accuracy": 0.52, "precision": 0.53, "recall": 0.37, "f1": 0.43, "pair_accuracy": 0.55})
    return runs, results
