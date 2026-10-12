# Extending CodeGuard

Components are looked up by name in three registries (`src/codeguard/registry.py`).
Adding one = write it, register it, reference it from a YAML config. The CLI,
the data pipeline and run management do not change.

| Registry | Lives in | Config field |
|---|---|---|
| `DATASETS` | `src/codeguard/data/sources/` | `source.name` in a data config |
| `MODELS` | `src/codeguard/models/` | `model.name` in a train config |
| `METRICS` | `src/codeguard/evaluation/metrics.py` | used by evaluation |

## Add a dataset (e.g. FixJS for JavaScript)

A source is a function `(SourceConfig) -> Iterator[dict]` that yields raw records with
at least `project`, `diff` and `sstub_pattern` (use `"UNKNOWN"` if absent). Optional
fields: `project_url`, `file_path`, `commit_sha`, `likely_bug`, `in_function`,
`before`, `after`.

```python
# src/codeguard/data/sources/fixjs.py
from codeguard.registry import DATASETS


@DATASETS.register("fixjs")
def read_fixjs(cfg):
    for row in _iter_raw_rows(cfg.path):  # your parsing
        yield {"project": row["repo"], "diff": row["diff"], "sstub_pattern": "UNKNOWN"}
```

1. Import the module in `src/codeguard/data/sources/__init__.py` (this registers it).
2. Copy a data config, set `source.name: fixjs`, and set record filters that do not apply
   to `null` (e.g. `in_function: null`).
3. `codeguard prepare-data -c configs/prototype/data_fixjs.yaml`.

Filtering, reconstruction, deduplication, grouped split, leakage report and manifest
work unchanged. Python 2 markers are Python-specific; they simply never fire on other
languages.

If a dataset already comes as buggy/fixed pairs instead of diffs (e.g. CodeXGLUE
refinement), have the reader build a minimal diff (`"@@ @@\n-<buggy>\n+<fixed>"`) or
add a `buggy`/`fixed` passthrough in `_pool_row`. That is the one place to touch.

## Add a model (implemented from stage 2)

Models implement the protocol in `src/codeguard/models/base.py` (`fit`,
`predict_proba`, `save`, `load`) and are registered with `@MODELS.register("name")`.
Constructor keyword arguments come from `model.params` in the train config. See
`src/codeguard/models/classical.py` for a minimal example.

## Add a metric (implemented from stage 2)

A metric is a function registered with `@METRICS.register("name")` in
`src/codeguard/evaluation/metrics.py`. If it should appear in the shared results table,
also add a column to `RESULT_COLUMNS` in `src/codeguard/utils/results.py` (the schema is
fixed on purpose, so that old rows stay comparable).

## Add a config tier

`tier` is a free label (`smoke`, `prototype`, `server`) written to every results row.
New tiers only need a folder under `configs/`, plus the value in `config.Tier`.
