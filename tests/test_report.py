"""make_report.py on fabricated run directories (tests/fixtures/report_runs.py); no real results involved."""

import json
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import make_report
from experiment_utils import paired_bootstrap_diff
from report_runs import make_runs


def report(tmp_path, test_rates=None, name="w"):
    runs, results = make_runs(tmp_path / name, test_rates)
    assert make_report.run(Namespace(runs=str(runs), results=str(results))) == 0
    return runs, results


def keys(data):
    if isinstance(data, dict):
        return set(data) | {k for v in data.values() for k in keys(v)}
    if isinstance(data, list):
        return {k for v in data for k in keys(v)}
    return set()


def test_selection_uses_val_even_when_another_run_is_best_on_test(tmp_path):
    runs, results = report(tmp_path)
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    assert selection["selected_run"] == "lora_6.7b_r16"  # best val among full-tier, non-replicate LoRA
    assert selection["tier"] == "full" and selection["criterion"] == "val exact_match"
    assert "lora_6.7b_r16_s43" not in selection["candidates"]  # seed replicate (best val) excluded
    assert "lora_1.3b_r16_small" not in selection["candidates"]  # small tier, since full runs exist
    table = pd.read_csv(results / "results.csv")
    with_test = [c for c in selection["candidates"] if c != "lora_6.7b_r32"]  # r32 has no test split
    test = table[table.split == "test"].set_index("run").loc[with_test]
    assert test.exact_match.idxmax() == "lora_6.7b_r8"  # a different run wins on test; selection ignores it


def test_selection_json_holds_no_test_numbers(tmp_path):
    _, results = report(tmp_path, name="a")
    text = (results / "selection.json").read_text(encoding="utf-8")
    selection = json.loads(text)
    assert "test" not in keys(selection) and '"test"' not in text
    assert {row["split"] for row in selection["val_table"]} == {"val"}
    val = pd.read_csv(results / "results.csv").query("split == 'val'").set_index("run")
    for row in selection["val_table"]:
        assert row["exact_match"] == val.loc[row["run"], "exact_match"]
    # Changing every test number leaves selection.json byte-identical.
    _, other = report(tmp_path, {name: 0.9 for name in ("lora_6.7b_r8", "zs_6.7b", "qlora_6.7b_r16")}, name="b")
    assert (other / "selection.json").read_text(encoding="utf-8") == text
    assert not pd.read_csv(other / "results.csv").equals(pd.read_csv(results / "results.csv"))


def test_results_csv_has_both_splits_and_marks_the_appendix(tmp_path):
    _, results = report(tmp_path)
    table = pd.read_csv(results / "results.csv")
    assert set(table.split) == {"val", "test"}
    rows = table.set_index(["run", "split"])
    assert ("lora_6.7b_r32", "val") in rows.index and ("lora_6.7b_r32", "test") not in rows.index
    headline = set(table[table.headline].run)
    assert headline == {"lora_6.7b_r16", "zs_6.7b", "fs3_6.7b", "zs_33b"}
    assert rows.loc[("lora_6.7b_r8", "test"), "section"] == "test appendix (not used for selection)"
    assert rows.loc[("lora_6.7b_r16", "test"), "section"] == "test headline"
    assert not table[table.split == "test"].used_for_selection.any()
    assert rows.loc[("lora_6.7b_r8", "val"), "used_for_selection"]
    assert not rows.loc[("zs_6.7b", "val"), "used_for_selection"]
    assert pd.isna(rows.loc[("lora_6.7b_r32", "val"), "rank"])  # missing fields stay empty
    assert pd.isna(rows.loc[("zs_6.7b", "val"), "trainable_parameters"])  # untrained run: not a cost


def test_paired_difference_uses_the_pair_id_intersection(tmp_path):
    runs, results = report(tmp_path)
    headline = json.loads((results / "headline.json").read_text(encoding="utf-8"))
    comparison = next(c for c in headline["comparisons"] if c["b"] == "zs_6.7b")
    assert (comparison["n_a"], comparison["n_b"], comparison["n"]) == (60, 50, 50)
    a = pd.read_parquet(runs / "lora_6.7b_r16" / "test_predictions.parquet")
    b = pd.read_parquet(runs / "zs_6.7b" / "test_predictions.parquet")
    joined = a.merge(b, on="pair_id", suffixes=("_a", "_b"))
    for metric in ("exact_match", "localized", "direction_correct"):
        expected = paired_bootstrap_diff(joined[f"{metric}_a"].astype(float), joined[f"{metric}_b"].astype(float))
        assert comparison[metric] == {**expected, "n": 50}
    assert {c["b"] for c in headline["comparisons"]} == {"zs_6.7b", "fs3_6.7b", "zs_33b"}
    assert [row["role"] for row in headline["rows"]] == ["selected", "zero-shot", "3-shot", "33b zero-shot"]


def test_by_pattern_detection_and_costs(tmp_path):
    _, results = report(tmp_path)
    patterns = pd.read_csv(results / "by_pattern.csv")
    assert set(patterns.sstub_pattern) == {"CHANGE_IDENTIFIER_USED", "OTHER"}
    assert set(patterns.role) == {"selected", "zero-shot", "3-shot", "33b zero-shot"}
    detection = pd.read_csv(results / "detection.csv")
    assert {"TF-IDF + logistic regression", "BiLSTM", "LLM zero-shot", "LLM LoRA", "fix direction"} <= set(detection.system)
    assert detection.set_index("run").loc["cls_zs_6.7b", "system"] == "LLM zero-shot"  # legacy, no task field
    costs = pd.read_csv(results / "costs.csv").set_index("run")
    assert costs.loc["lora_6.7b_r16", "trainable_share"] == pytest.approx(16 * 2_500_000 / 6_700_000_000)
    assert pd.isna(costs.loc["zs_6.7b", "trainable_parameters"])
    assert pd.isna(costs.loc["lora_6.7b_r32", "adapter_mb"])


def test_ties_prefer_fewer_trainable_parameters_then_name():
    def row(name, em, params):
        return {"run": name, "tier": "full", "method": "lora", "exact_match": em, "trainable_parameters": params}

    rows = [row("lora_b", 0.3, 20), row("lora_a", 0.3, 20), row("lora_c", 0.3, 10), row("lora_d", 0.2, 1)]
    assert make_report.select_final(rows)["selected_run"] == "lora_c"
    assert make_report.select_final(rows[:2])["selected_run"] == "lora_a"
    assert make_report.select_final([{**rows[0], "method": "zero-shot"}])["selected_run"] is None


def test_no_runs_exits_cleanly(tmp_path):
    results = tmp_path / "results"
    assert make_report.run(Namespace(runs=str(tmp_path / "missing"), results=str(results))) == 0
    (tmp_path / "empty" / "unfinished").mkdir(parents=True)
    done = subprocess.run([sys.executable, str(ROOT / "src" / "make_report.py"), "--runs", str(tmp_path / "empty"),
                           "--results", str(results)], capture_output=True, text=True)
    assert done.returncode == 0 and "nothing to report" in done.stdout
    assert not results.exists()


def test_only_zero_shot_runs_select_nothing(tmp_path):
    runs, results = make_runs(tmp_path)
    for path in runs.iterdir():
        if "lora" in path.name:
            (path / "metrics.json").unlink(missing_ok=True)
    assert make_report.run(Namespace(runs=str(runs), results=str(results))) == 0
    assert json.loads((results / "selection.json").read_text(encoding="utf-8"))["selected_run"] is None
    assert json.loads((results / "headline.json").read_text(encoding="utf-8"))["rows"] == []
    assert pd.read_csv(results / "results.csv").headline.sum() == 0


def test_report_does_not_import_torch():
    code = "import sys; sys.path.insert(0, 'src'); import make_report; print('torch' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "False"


def test_missing_localized_exact_interval_is_computed_from_predictions(tmp_path):
    from experiment_utils import bootstrap_ci

    runs, results = report(tmp_path)  # fixture metrics.json files have no localized_exact interval
    for split in ("val", "test"):
        predictions = pd.read_parquet(runs / "lora_1.3b_r16_small" / f"{split}_predictions.parquet")
        expected = bootstrap_ci(predictions["localized_exact"].astype(float))
        table = pd.read_csv(results / "results.csv")
        row = table[(table["run"] == "lora_1.3b_r16_small") & (table["split"] == split)].iloc[0]
        assert row["localized_exact_low"] == pytest.approx(expected["low"])
        assert row["localized_exact_high"] == pytest.approx(expected["high"])
    headline = json.loads((results / "headline.json").read_text(encoding="utf-8"))
    assert all("localized_exact" in row for row in headline["rows"])
    assert "localized_exact" in headline["comparisons"][0]
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    assert "localized_exact_low" in selection["val_table"][0]


def test_runs_with_the_old_generation_rule_are_only_listed_as_superseded(tmp_path):
    runs, results = make_runs(tmp_path / "w")
    old = runs / "zs_1.3b_small_truncated_v1"
    shutil.copytree(runs / "zs_1.3b_small", old)
    metrics = json.loads((old / "metrics.json").read_text(encoding="utf-8"))
    del metrics["generation"]
    (old / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert make_report.run(Namespace(runs=str(runs), results=str(results))) == 0
    superseded = pd.read_csv(results / "superseded.csv")
    assert set(superseded["run"]) == {"zs_1.3b_small_truncated_v1"} and set(superseded["split"]) == {"val", "test"}
    assert "zs_1.3b_small_truncated_v1" not in set(pd.read_csv(results / "results.csv")["run"])
    selection = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    assert "zs_1.3b_small_truncated_v1" not in {row["run"] for row in selection["val_table"]}
