"""Typed experiment configs, loaded from YAML and validated with pydantic.

Unknown keys are rejected, so a typo in a config fails loudly instead of being
silently ignored. Any field can be overridden from the CLI with
``-o dotted.key=value`` (the value is parsed as YAML, so ``-o seed=7`` is an int).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Tier = Literal["smoke", "prototype", "server"]
Task = Literal["classification", "generation"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- data


class SourceConfig(StrictModel):
    name: str = "ctssb"  # key in codeguard.registry.DATASETS
    path: Path
    max_rows: int | None = Field(default=None, ge=1)  # read only the first N raw rows (smoke)


class FilterConfig(StrictModel):
    likely_bug: bool | None = True  # None = do not filter on this field
    in_function: bool | None = True
    single_hunk: bool = True
    drop_python2: bool = False


class SplitConfig(StrictModel):
    group_key: str = "project"
    val: float = Field(default=0.1, gt=0, lt=1)
    test: float = Field(default=0.1, gt=0, lt=1)
    drop_cross_split_duplicates: bool = True

    @model_validator(mode="after")
    def _fractions_leave_room_for_train(self) -> SplitConfig:
        if self.val + self.test >= 1:
            raise ValueError("val + test must be < 1")
        return self


class SampleConfig(StrictModel):
    n_pairs: int | None = Field(default=None, ge=1)  # None = keep every pair
    max_pairs_per_project: int | None = Field(default=None, ge=1)


class DataConfig(StrictModel):
    name: str
    tier: Tier = "prototype"
    source: SourceConfig
    filters: FilterConfig = FilterConfig()
    split: SplitConfig = SplitConfig()
    sample: SampleConfig = SampleConfig()
    seed: int = 42
    output_dir: Path
    cache_dir: Path = Path("data/processed/_cache")


# ---------------------------------------------------------------------- training


class ModelSpec(StrictModel):
    name: str  # key in codeguard.registry.MODELS
    params: dict[str, Any] = {}


class TuningConfig(StrictModel):
    """Grid search over one hyper-parameter, selected on the validation split."""

    param: str
    grid: list[Any] = Field(min_length=1)
    metric: str = "f1"


class TrainConfig(StrictModel):
    name: str
    tier: Tier = "prototype"
    task: Task = "classification"
    data: Path  # directory produced by `codeguard prepare-data`
    model: ModelSpec
    tuning: TuningConfig | None = None
    seed: int = 42
    runs_dir: Path = Path("runs")
    results_csv: Path = Path("results/results.csv")


# ----------------------------------------------------------------------- loading

C = TypeVar("C", bound=BaseModel)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides in place; values are parsed as YAML."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        node = raw
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot override '{key}': '{part}' is not a mapping")
        node[leaf] = yaml.safe_load(value)
    return raw


def load_config(path: Path, schema: type[C], overrides: list[str] | None = None) -> C:
    """Read a YAML file, apply CLI overrides and validate it against ``schema``."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return schema.model_validate(apply_overrides(raw, overrides or []))


def dump_config(config: BaseModel, path: Path) -> None:
    """Write the fully resolved config (defaults included) as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)
