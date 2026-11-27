"""Collect runs/*/metrics.json into the report tables and pick the final model on val only.

Order (docs/AGENT_GUIDE.md section 6.5):
  1. read every run's metrics.json;
  2. build the val table and choose the final LoRA run by val exact match;
  3. write results/selection.json (the choice and the val table it used);
  4. only then read test numbers: results.csv, headline.json, by_pattern.csv, detection.csv, costs.csv.

A run counts only if <runs dir>/<name>/metrics.json exists; several runs directories may be given
(a name found in more than one is taken from the first). A missing field is written as an empty cell.

Runs are compared only on the pair_ids all of them predicted: results.csv carries every run's own
numbers (n) and its numbers on the pairs shared by all current runs of that split (n_shared, *_shared);
selection compares the candidates on the val pairs all candidates predicted; paired differences and
by_pattern.csv use the pairs the compared runs share, with the count next to every number. Runs
evaluated before the prefilled-fence generation rule are written to superseded.csv only.
Needs pandas/numpy only (no torch).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from experiment_utils import bootstrap_ci, paired_bootstrap_diff, write_json
from repair_utils import GENERATION_RULE, MIN_PATTERN_PAIRS

TRAINED = {"lora", "qlora"}
REPLICATE = re.compile(r"_s\d+$")  # lora_6.7b_r16_s43 / _s44: seed replicates, never selected
TIER_PREFERENCE = ("full", "small")
METRICS = ("n", "exact_match", "changed", "localized", "localized_exact", "parse_ok", "direction_correct",
           "bleu", "copy_bleu", "copy_exact_match", "localization_chance", "direction_chance")
INTERVALS = ("exact_match", "localized", "localized_exact", "direction_correct")
PAIRED = INTERVALS
SHARED = ("exact_match", "localized_exact", "localized", "parse_ok")  # also computed on the shared pair_ids
DESCRIBE = ("run", "tier", "model", "method", "rank", "quantized")
RESULT_COLUMNS = (["section", "run", "split", "tier", "model", "method", "rank", "quantized", "used_for_selection",
                   "selected", "headline", *METRICS]
                  + [f"{m}_{end}" for m in INTERVALS for end in ("low", "high")]
                  + ["n_shared"] + [f"{m}_shared{end}" for m in SHARED for end in ("", "_low", "_high")]
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


def load_runs(runs_dirs):
    """Every <runs dir>/<name>/metrics.json, with the run name taken from the directory. runs_dirs is
    one directory or a list; a name found in more than one is taken from the first. Returns the runs
    and {run name: the runs directory it was read from}."""
    runs, seen = [], {}
    for runs_dir in ([runs_dirs] if isinstance(runs_dirs, (str, Path)) else runs_dirs):
        for path in sorted(Path(runs_dir).glob("*/metrics.json")):
            name = path.parent.name
            if name in seen:
                print(f"skipping {path}: {name} already read from {seen[name]}")
                continue
            try:
                metrics = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                print(f"skipping {path}: {error}")
                continue
            metrics["run"], seen[name] = name, path.parent.parent
            runs.append(metrics)
    return runs, seen


def task(run):
    if run.get("task"):
        return run["task"]
    return "detection" if run["run"].startswith(("cls_", "bilstm")) else "repair"


def is_current(run):
    """Repair runs evaluated with the current generation rule (prefilled fence, stop at the closing
    fence). Older runs let the model explain first and were cut off by the token budget; they are
    reported only in superseded.csv, never compared or selected."""
    return task(run) != "repair" or run.get("generation") == GENERATION_RULE


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


def select_final(val_rows, locations=None):
    """Highest val exact match among non-replicate LoRA/QLoRA runs of the preferred tier.

    Receives the val table only. With locations (run name -> runs dir) and a val prediction file for
    every candidate, the candidates are compared on the val pair_ids all of them predicted.
    Ties: fewer trainable parameters, then run name.
    """
    candidates = [row for row in val_rows if is_candidate(row)]
    tiers = {row["tier"] for row in candidates}
    chosen_tier = next((t for t in TIER_PREFERENCE if t in tiers), min(tiers, key=str) if tiers else None)
    pool = [dict(row) for row in candidates if row["tier"] == chosen_tier]
    shared = None
    if locations is not None and pool:
        tables = {row["run"]: read_predictions(locations, row["run"], "val") for row in pool}
        if all(t is not None for t in tables.values()):
            shared = set.intersection(*(set(t["pair_id"]) for t in tables.values()))
            for row in pool:
                table = tables[row["run"]]
                row["exact_match"] = float(table[table["pair_id"].isin(shared)]["exact_match"].astype(float).mean())
    pool.sort(key=lambda row: (-row["exact_match"], row["trainable_parameters"] or float("inf"), row["run"]))
    return {
        "selected_run": pool[0]["run"] if pool else None,
        "criterion": "val exact_match" + (" on the val pair_ids every candidate predicted" if shared else ""),
        "n_shared_val_pairs": len(shared) if shared else None,
        "candidate_val_exact_match": {row["run"]: row["exact_match"] for row in pool},
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


def shared_numbers(repair_runs, locations, split):
    """{run: {n_shared, metric_shared, metric_shared_low/high}} on the pair_ids that every current run
    with a prediction file for `split` predicted."""
    tables = {run["run"]: read_predictions(locations, run["run"], split) for run in repair_runs}
    tables = {name: table for name, table in tables.items() if table is not None}
    if not tables:
        return {}
    shared = set.intersection(*(set(t["pair_id"]) for t in tables.values()))
    out = {}
    for name, table in tables.items():
        part = table[table["pair_id"].isin(shared)]
        row = {"n_shared": len(part)}
        for metric in SHARED:
            if metric in part.columns and len(part):
                interval = bootstrap_ci(part[metric].astype(float))
                row.update({f"{metric}_shared": interval["mean"], f"{metric}_shared_low": interval["low"],
                            f"{metric}_shared_high": interval["high"]})
        out[name] = row
    return out


def results_table(repair_runs, selection, headline, locations=None):
    headline_names = {run["run"] for _, run in headline}
    rows = []
    for split in ("val", "test"):
        shared = shared_numbers(repair_runs, locations, split) if locations is not None else {}
        for run in repair_runs:
            if get(run, "splits", split) is None:
                continue
            row = split_row(run, split)
            row.update(shared.get(run["run"], {}))
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


def superseded_table(runs):
    """Runs evaluated with the old generation rule: their own numbers, for the report's lesson only."""
    rows = [{**split_row(run, split), "generation": run.get("generation") or "v1 (no prefill, truncated)"}
            for run in runs for split in ("val", "test") if get(run, "splits", split) is not None]
    columns = ["run", "split", "model", "method", "generation", "n", "exact_match", "exact_match_low",
               "exact_match_high", "localized_exact", "parse_ok", "direction_correct"]
    return pd.DataFrame(rows, columns=columns)


def read_predictions(runs_dir, name, split):
    """runs_dir: one directory, or a dict run name -> the runs directory that holds it."""
    if isinstance(runs_dir, dict):
        if name not in runs_dir:
            return None
        runs_dir = runs_dir[name]
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


def by_pattern_table(headline, locations):
    """Per bug type on test, on the pair_ids every headline run predicted (bug types with fewer than
    MIN_PATTERN_PAIRS shared pairs grouped as OTHER); n is the number of shared pairs of that type."""
    columns = ["role", "run", "split", "sstub_pattern", "n", "exact_match", "exact_match_low", "exact_match_high",
               "localized", "localized_exact"]
    tables = [(role, run["run"], read_predictions(locations, run["run"], "test")) for role, run in headline]
    tables = [(role, name, table) for role, name, table in tables if table is not None]
    if not tables:
        return pd.DataFrame([], columns=columns)
    shared = set.intersection(*(set(t["pair_id"]) for _, _, t in tables))
    rows = []
    for role, name, table in tables:
        part = table[table["pair_id"].isin(shared)]
        counts = part["sstub_pattern"].value_counts()
        pattern = part["sstub_pattern"].where(part["sstub_pattern"].map(counts) >= MIN_PATTERN_PAIRS, "OTHER")
        for value, group in part.groupby(pattern):
            interval = bootstrap_ci(group["exact_match"].astype(float))
            rows.append({"role": role, "run": name, "split": "test", "sstub_pattern": value, "n": len(group),
                         "exact_match": interval["mean"], "exact_match_low": interval["low"],
                         "exact_match_high": interval["high"],
                         "localized": float(group["localized"].astype(float).mean()),
                         "localized_exact": float(group["localized_exact"].astype(float).mean())})
    return pd.DataFrame(rows, columns=columns)


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
    runs, locations = load_runs(args.runs)
    if not runs:
        print(f"No runs with metrics.json under {args.runs}; nothing to report.")
        return 0
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    superseded = [r for r in runs if task(r) == "repair" and not is_current(r)]
    repair = [r for r in runs if task(r) == "repair" and is_current(r)]
    detection = [r for r in runs if task(r) == "detection"]
    superseded_table(superseded).to_csv(results / "superseded.csv", index=False)
    if superseded:
        print("Superseded (old generation rule, not compared): " + ", ".join(r["run"] for r in superseded))

    # Selection: val only, written to disk before any test number is read.
    fill_missing_intervals(repair, locations, "val")
    val_rows = val_table(repair)
    selection = select_final(val_rows, locations)
    write_json(results / "selection.json", selection)
    print_val_table(val_rows, selection)

    # Test numbers from here on.
    fill_missing_intervals(repair, locations, "test")
    headline = headline_runs(repair, selection)
    results_table(repair, selection, headline, locations).to_csv(results / "results.csv", index=False)
    write_json(results / "headline.json", headline_summary(locations, headline, selection))
    by_pattern_table(headline, locations).to_csv(results / "by_pattern.csv", index=False)
    detection_table(detection, repair, results).to_csv(results / "detection.csv", index=False)
    costs_table(repair).to_csv(results / "costs.csv", index=False)
    print(f"\n{len(repair)} repair and {len(detection)} detection runs; tables written to {results}")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=["runs"],
                   help="one or more directories with one subdirectory per run (the first wins on a name clash)")
    p.add_argument("--results", default="results", help="where the tables and selection.json are written")
    return p


if __name__ == "__main__":
    sys.exit(run(parser().parse_args()))
