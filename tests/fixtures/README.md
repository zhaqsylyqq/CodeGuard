# Test fixtures (fake data)

`report_runs.py` writes **fabricated** run directories (`runs/<name>/metrics.json`, prediction
parquet files) and a fake `baseline_metrics.json` into a temporary directory for
`tests/test_report.py` and for checking `notebooks/06_results.ipynb`.

Every number in them is invented to exercise the report code. They are not results. Never copy
them into `runs/` or `results/`; every generated `metrics.json` carries a `"fixture"` field saying so.
