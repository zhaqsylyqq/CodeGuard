"""--multi-gpu (33B over two GPUs, one process) and the preflight 33B check that must never fail preflight."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")


def _args(*extra):
    from train_repair import parser

    return parser().parse_args(["--output", "unused", *extra])


def test_multi_gpu_needs_quantized(monkeypatch):
    from train_repair import check_gpu_mode

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(SystemExit, match="--quantized"):
        check_gpu_mode(_args("--multi-gpu"))


def test_single_gpu_guard_stays_without_multi_gpu(monkeypatch):
    from train_repair import check_gpu_mode

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(SystemExit, match="CUDA_VISIBLE_DEVICES"):
        check_gpu_mode(_args("--quantized"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    check_gpu_mode(_args())  # one visible GPU: the normal 6.7B path is unchanged


def test_multi_gpu_lifts_the_guard_for_a_quantized_model(monkeypatch):
    from train_repair import check_gpu_mode

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    check_gpu_mode(_args("--quantized", "--multi-gpu"))


def test_installed_trainer_never_wraps_a_4bit_or_split_model_in_dataparallel():
    """Checked against the installed transformers source, as the design asks."""
    import inspect

    from transformers import Trainer

    wrap = inspect.getsource(Trainer._wrap_model)
    assert 'not getattr(model, "is_loaded_in_4bit", False)' in wrap
    init = inspect.getsource(Trainer)
    assert "self.is_model_parallel = True" in init and "self.args._n_gpu = 1" in init


def test_failed_33b_check_only_warns(tmp_path, monkeypatch, capsys):
    import preflight

    monkeypatch.setattr(preflight, "problems", [])
    args = preflight.parser().parse_args(["--data-dir", str(tmp_path), "--window-min", "210"])
    result = preflight.check_33b("no-such-org/no-such-model", ["0", "1"], args)
    assert result["ok"] is False and result["fits_window"] is False
    assert preflight.problems == []
    assert "WITH_33B=0" in capsys.readouterr().out


def test_timing_file_merges_measurements(tmp_path):
    import preflight

    path = tmp_path / "logs" / "preflight_timing.json"
    preflight.save_timing(path, "main", {"examples_per_second": 3.0})
    preflight.save_timing(path, "33b", {"ok": False})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["main"]["examples_per_second"] == 3.0 and data["33b"]["ok"] is False
