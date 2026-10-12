from pathlib import Path

import pytest
from pydantic import ValidationError

from codeguard.config import DataConfig, TrainConfig, apply_overrides, load_config
from codeguard.registry import Registry

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_overrides_parse_yaml_values_and_create_nested_keys():
    raw = {"model": {"params": {"C": 1.0}}}
    apply_overrides(raw, ["model.params.C=10", "seed=7", "tuning.grid=[0.1, 1]"])
    assert raw["model"]["params"]["C"] == 10
    assert raw["seed"] == 7
    assert raw["tuning"]["grid"] == [0.1, 1]


def test_malformed_override_is_rejected():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(
            {"name": "x", "data": "d", "model": {"name": "dummy"}, "lerning_rate": 1}
        )


def test_split_fractions_must_leave_room_for_train():
    with pytest.raises(ValidationError):
        DataConfig.model_validate(
            {
                "name": "x",
                "source": {"path": "p"},
                "output_dir": "o",
                "split": {"val": 0.5, "test": 0.5},
            }
        )


@pytest.mark.parametrize("path", sorted(CONFIG_ROOT.glob("*/*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_validate(path: Path):
    schema = DataConfig if path.name.startswith("data") else TrainConfig
    load_config(path, schema)


def test_registry_register_get_and_duplicates():
    reg: Registry = Registry("thing")

    @reg.register("a")
    class A:
        pass

    assert reg.get("a") is A
    assert "a" in reg and reg.names() == ["a"]
    with pytest.raises(ValueError):
        reg.register("a")(A)
    with pytest.raises(KeyError, match="Available: a"):
        reg.get("b")
