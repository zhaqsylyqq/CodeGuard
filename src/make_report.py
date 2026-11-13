"""Collect runs/*/metrics.json into the report tables and pick the final model on val only.

Order (docs/AGENT_GUIDE.md section 6.5):
  1. read every run's metrics.json;
  2. build the val table and choose the final LoRA run by val exact match;
  3. write results/selection.json (the choice and the val table it used);
  4. only then read test numbers: results.csv, headline.json, by_pattern.csv, detection.csv, costs.csv.

A run counts only if runs/<name>/metrics.json exists. A missing field is written as an empty cell.
Needs pandas/numpy only (no torch).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from experiment_utils import bootstrap_ci, paired_bootstrap_diff, write_json

TRAINED = {"lora", "qlora"}
REPLICATE = re.compile(r"_s\d+$")  # lora_6.7b_r16_s43 / _s44: seed replicates, never selected
TIER_PREFERENCE = ("full", "small")
METRICS = ("n", "exact_match", "changed", "localized", "localized_exact", "parse_ok", "direction_correct",
           "bleu", "copy_bleu", "copy_exact_match", "localization_chance", "direction_chance")
INTERVALS = ("exact_match", "localized", "localized_exact", "direction_correct")
PAIRED = INTERVALS
DESCRIBE = ("run", "tier", "model", "method", "rank", "quantized")
RESULT_COLUMNS = (["section", "run", "split", "tier", "model", "method", "rank", "quantized", "used_for_selection",
                   "selected", "headline", *METRICS]
                  + [f"{m}_{end}" for m in INTERVALS for end in ("low", "high")]
                  + ["eval_seconds", "pairs_per_second", "trainable_parameters", "total_parameters", "adapter_bytes",
                     "train_seconds", "train_samples_per_second"])
COSTS = ("trainable_parameters", "total_parameters", "adapter_bytes", "train_seconds",
         "train_samples_per_second")


def get(data, *keys):
    """data[k1][k2]... or None if any level is missing."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


# ---------------------------------------------------------------- 1. runs


def load_runs(runs_dir):
    """Every runs/<name>/metrics.json, with the run name taken from the directory."""
    runs = []
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"skipping {path}: {error}")
            continue
        metrics["run"] = path.parent.name
        runs.append(metrics)
    return runs


def task(run):
    if run.get("task"):
        return run["task"]
    return "detection" if run["run"].startswith(("cls_", "bilstm")) else "repair"


def tier(run):
    data_dir = get(run, "args", "data_dir") or get(run, "settings", "data_dir")
    return Path(str(data_dir)).name if data_dir else None


def describe(run):
    return {"run": run["run"], "tier": tier(run), "model": run.get("model"), "method": run.get("method"),
            "rank": run.get("rank"), "quantized": run.get("quantized")}


def split_row(run, split):
    """Overall metrics and intervals of one split of one repair run (None where missing)."""
    row = describe(run)
    row["split"] = split
    for name in METRICS:
        row[name] = get(run, "splits", split, "overall", name)
    for name in INTERVALS:
        row[f"{name}_low"] = get(run, "splits", split, "intervals", name, "low")
        row[f"{name}_high"] = get(run, "splits", split, "intervals", name, "high")
    row["eval_seconds"] = get(run, "splits", split, "eval_seconds")
    row["pairs_per_second"] = get(run, "splits", split, "pairs_per_second")
    return row


# ---------------------------------------------------------------- 2-3. selection on val only


def val_table(repair_runs):
    """One row per repair run, val split only. Nothing from test is touched here."""
    rows = []
    for run in repair_runs:
        row = split_row(run, "val")
        # parameter count is a tie-breaker; it is only meaningful for runs that were trained
        row["trainable_parameters"] = run.get("trainable_parameters") if run.get("train_seconds") is not None else None
        rows.append(row)
    return rows


def is_candidate(row):
    return (row["method"] in TRAINED and not REPLICATE.search(row["run"])
            and isinstance(row["exact_match"], (int, float)))


def select_final(val_rows):
    """Highest val exact match among non-replicate LoRA/QLoRA runs of the preferred tier.

    Receives the val table only. Ties: fewer trainable parameters, then run name.
    """
    candidates = [row for row in val_rows if is_candidate(row)]
    tiers = {row["tier"] for row in candidates}
    chosen_tier = next((t for t in TIER_PREFERENCE if t in tiers), min(tiers, key=str) if tiers else None)
    pool = [row for row in candidates if row["tier"] == chosen_tier]
    pool.sort(key=lambda row: (-row["exact_match"], row["trainable_parameters"] or float("inf"), row["run"]))
    return {
        "selected_run": pool[0]["run"] if pool else None,
        "criterion": "val exact_match",
        "tie_breakers": ["fewer trainable parameters", "run name"],
        "tier": chosen_tier,
        "candidates": [row["run"] for row in pool],
        "val_table": val_rows,
        "note": "test was not read before this file was written",
    }


# ---------------------------------------------------------------- 4. test (after selection.json exists)


def find_baseline(repair_runs, selected, method):
    """Run with `method` on the selected run's model; same tier preferred, then name."""
    matches = [r for r in repair_runs if r.get("method") == method and r.get("model") == selected.get("model")]
    matches.sort(key=lambda r: (tier(r) != tier(selected), r["run"]))
    return matches[0] if matches else None


def headline_runs(repair_runs, selection):
    """[(role, run)] shown with test numbers: selected LoRA, its zero-shot and 3-shot, 33B zero-shot."""
    by_name = {r["run"]: r for r in repair_runs}
    selected = by_name.get(selection["selected_run"])
    if selected is None:
        return []
    roles = [("selected", selected), ("zero-shot", find_baseline(repair_runs, selected, "zero-shot")),
             ("3-shot", find_baseline(repair_runs, selected, "3-shot")), ("33b zero-shot", by_name.get("zs_33b"))]
    return [(role, run) for role, run in roles if run is not None]


def results_table(repair_runs, selection, headline):
    headline_names = {run["run"] for _, run in headline}
    rows = []
    for split in ("val", "test"):
        for run in repair_runs:
            if get(run, "splits", split) is None:
                continue
            row = split_row(run, split)
            trained = run.get("train_seconds") is not None
            for name in COSTS:
                row[name] = run.get(name) if trained or name == "total_parameters" else None
            row["used_for_selection"] = split == "val" and run["run"] in selection["candidates"]
            row["selected"] = run["run"] == selection["selected_run"]
            row["headline"] = split == "test" and run["run"] in headline_names
            row["section"] = ("val" if split == "val" else
                              "test headline" if row["headline"] else "test appendix (not used for selection)")
            rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def read_predictions(runs_dir, name, split):
    path = Path(runs_dir) / name / f"{split}_predictions.parquet"
    return pd.read_parquet(path) if path.exists() else None


def fill_missing_intervals(runs, runs_dir, split):
    """Intervals an older metrics.json lacks (localized_exact was added later), computed from that run's
    prediction file with the same bootstrap and seed as repair_utils.summarize. Only `split` is read."""
    for run in runs:
        result = get(run, "splits", split)
        if not isinstance(result, dict):
            continue
        intervals = result.setdefault("intervals", {})
        missing = [metric for metric in INTERVALS if metric not in intervals]
        predictions = read_predictions(runs_dir, run["run"], split) if missing else None
        if predictions is None:
            continue
        for metric in missing:
            if metric in predictions.columns:
                values = predictions[metric].dropna()
                if len(values):
                    intervals[metric] = bootstrap_ci(values.astype(float))


def paired_difference(runs_dir, a, b, split="test"):
    """Paired bootstrap of a - b on the pair_ids both runs predicted."""
    left, right = read_predictions(runs_dir, a, split), read_predictions(runs_dir, b, split)
    out = {"a": a, "b": b, "split": split}
    if left is None or right is None:
        out["missing"] = f"{split}_predictions.parquet missing for {a if left is None else b}"
        return out
    joined = left.merge(right, on="pair_id", suffixes=("_a", "_b"))
    out.update({"n_a": len(left), "n_b": len(right), "n": len(joined)})
    for metric in PAIRED:
        columns = [f"{metric}_a", f"{metric}_b"]
        if not set(columns) <= set(joined.columns):
            continue
        both = joined[columns].dropna()
        if len(both):
            out[metric] = {**paired_bootstrap_diff(both[columns[0]].astype(float), both[columns[1]].astype(float)),
                           "n": len(both)}
    return out


def headline_summary(runs_dir, headline, selection):
    rows = []
    for role, run in headline:
        row = {"role": role, **describe(run), "n": get(run, "splits", "test", "overall", "n")}
        for metric in INTERVALS:
            interval = get(run, "splits", "test", "intervals", metric)
            mean = get(run, "splits", "test", "overall", metric)
            if interval or mean is not None:
                row[metric] = interval or {"mean": mean}
        rows.append(row)
    selected = selection["selected_run"]
    comparisons = [{"label": f"selected vs {role}", **paired_difference(runs_dir, selected, run["run"])}
                   for role, run in headline if role != "selected"]
    return {"selected_run": selected, "tier": selection["tier"], "split": "test",
            "difference": "selected minus other, paired bootstrap over shared pair_ids (2000 resamples, seed 42)",
            "rows": rows, "comparisons": comparisons}


def by_pattern_table(headline):
    """Per bug type on test, as summarize() stored it (patterns under 30 pairs grouped as OTHER)."""
    rows = []
    for role, run in headline:
        for pattern, rates in (get(run, "splits", "test", "by_pattern") or {}).items():
            rows.append({"role": role, "run": run["run"], "split": "test", "sstub_pattern": pattern,
                         "n": rates.get("n"), "exact_match": rates.get("exact_match"),
                         "localized": rates.get("localized")})
    return pd.DataFrame(rows, columns=["role", "run", "split", "sstub_pattern", "n", "exact_match", "localized"])


def detection_system(run):
    if "bilstm" in str(run.get("model")).lower() or run["run"].startswith("bilstm"):
        return "BiLSTM"
    if run.get("adapter") or run.get("method") in TRAINED:
        return "LLM LoRA"
    return "LLM zero-shot"


def detection_table(detection_runs, repair_runs, results_dir):
    rows = []
    baseline = Path(results_dir) / "baseline_metrics.json"
    if baseline.exists():
        tfidf = json.loads(baseline.read_text(encoding="utf-8"))
        rows.append({"system": "TF-IDF + logistic regression", "run": baseline.name, "kind": "isolated snippet",
                     "split": "val", **{k: tfidf.get(k) for k in ("accuracy", "precision", "recall", "f1",
                                                                  "pair_accuracy")}})
    for run in detection_runs:
        rows.append({"system": detection_system(run), **describe(run), "kind": "isolated snippet",
                     "split": run.get("split", "val"), "n": run.get("n_examples"),
                     **{k: run.get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}})
    for run in repair_runs:
        value = get(run, "splits", "val", "overall", "direction_correct")
        if value is None:
            continue
        rows.append({"system": "fix direction", **describe(run), "kind": "pair (fix direction)", "split": "val",
                     "n": get(run, "splits", "val", "overall", "n"), "accuracy": value,
                     "accuracy_low": get(run, "splits", "val", "intervals", "direction_correct", "low"),
                     "accuracy_high": get(run, "splits", "val", "intervals", "direction_correct", "high")})
    columns = ["system", "run", "kind", "tier", "model", "method", "rank", "split", "n", "accuracy",
               "accuracy_low", "accuracy_high", "precision", "recall", "f1", "roc_auc", "pair_accuracy"]
    return pd.DataFrame(rows, columns=columns)


def costs_table(repair_runs):
    rows = []
    for run in repair_runs:
        trained = run.get("train_seconds") is not None
        trainable = run.get("trainable_parameters") if trained else None
        total = run.get("total_parameters")
        size = run.get("adapter_bytes")
        seconds = run.get("train_seconds")
        rows.append({**describe(run), "trained": trained, "trainable_parameters": trainable,
                     "total_parameters": total,
                     "trainable_share": trainable / total if trainable and total else None,
                     "adapter_mb": size / 2**20 if size else None,
                     "train_minutes": seconds / 60 if seconds else None,
                     "train_examples": run.get("train_examples"),
                     "train_samples_per_second": run.get("train_samples_per_second") if trained else None,
                     "val_pairs_per_second": get(run, "splits", "val", "pairs_per_second"),
                     "stopped_by_time_budget": run.get("stopped_by_time_budget")})
    columns = [*DESCRIBE, "trained", "trainable_parameters", "total_parameters", "trainable_share", "adapter_mb",
               "train_minutes", "train_examples", "train_samples_per_second", "val_pairs_per_second",
               "stopped_by_time_budget"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------- output


def print_val_table(val_rows, selection):
    if not val_rows:
        print("No repair runs.")
        return
    table = pd.DataFrame(val_rows)
    table["rank"] = table["rank"].astype("Int64")
    em = table["exact_match"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    low, high = table["exact_match_low"], table["exact_match_high"]
    table["exact_match [95% CI]"] = em + [
        "" if pd.isna(a) or pd.isna(b) else f" [{a:.3f}, {b:.3f}]" for a, b in zip(low, high)]
    shown = ["run", "tier", "method", "rank", "n", "exact_match [95% CI]", "localized", "localized_exact",
             "direction_correct", "parse_ok", "changed"]
    table = table[shown].sort_values(["tier", "run"], na_position="last")
    print("Val table (repair):")
    print(table.to_string(index=False, na_rep="", float_format=lambda v: f"{v:.3f}"))
    if selection["selected_run"]:
        print(f"\nSelected on val ({selection['tier']} tier, by val exact match): {selection['selected_run']}")
    else:
        print("\nNo LoRA/QLoRA run with val exact match yet; nothing selected.")


def run(args):
    runs = load_runs(args.runs)
    if not runs:
        print(f"No runs with metrics.json under {args.runs}; nothing to report.")
        return 0
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    repair = [r for r in runs if task(r) == "repair"]
    detection = [r for r in runs if task(r) == "detection"]

    # Selection: val only, written to disk before any test number is read.
    fill_missing_intervals(repair, args.runs, "val")
    val_rows = val_table(repair)
    selection = select_final(val_rows)
    write_json(results / "selection.json", selection)
    print_val_table(val_rows, selection)

    # Test numbers from here on.
    fill_missing_intervals(repair, args.runs, "test")
    headline = headline_runs(repair, selection)
    results_table(repair, selection, headline).to_csv(results / "results.csv", index=False)
    write_json(results / "headline.json", headline_summary(args.runs, headline, selection))
    by_pattern_table(headline).to_csv(results / "by_pattern.csv", index=False)
    detection_table(detection, repair, results).to_csv(results / "detection.csv", index=False)
    costs_table(repair).to_csv(results / "costs.csv", index=False)
    print(f"\n{len(repair)} repair and {len(detection)} detection runs; tables written to {results}")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs", help="directory with one subdirectory per run")
    p.add_argument("--results", default="results", help="where the tables and selection.json are written")
    return p


if __name__ == "__main__":
    sys.exit(run(parser().parse_args()))
