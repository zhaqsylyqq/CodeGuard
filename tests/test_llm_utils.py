"""Device / dtype choice, time budget and parameter counting (no downloads)."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_utils import resolve_dtype


def test_auto_dtype_ignores_emulated_bf16(monkeypatch):
    # A T4 (compute capability 7.5) reports bf16 only "with emulation"; auto must pick fp16 there.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: including_emulation)
    assert resolve_dtype("auto") == torch.float16


def test_auto_dtype_uses_native_bf16_and_cpu_fp32(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)
    assert resolve_dtype("auto") == torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_dtype("auto") == torch.float32
    assert resolve_dtype("fp16") == torch.float16


# ---------------------------------------------------------------- time budget across restarts


def _budget_step(callback, step=1):
    from transformers import TrainerControl, TrainerState, TrainingArguments

    args = TrainingArguments(output_dir="unused", eval_strategy="steps", save_strategy="steps", report_to="none")
    state = TrainerState()
    state.global_step = step
    return callback.on_step_end(args, state, TrainerControl())


def test_time_budget_writes_elapsed_time(tmp_path):
    from llm_utils import TimeBudgetCallback

    state_file = tmp_path / "train_time.json"
    callback = TimeBudgetCallback(10, state_file)
    callback.on_train_begin(None, None, None)
    control = _budget_step(callback, step=3)
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["step"] == 3 and 0 <= saved["elapsed_seconds"] < 60
    assert not control.should_training_stop and not callback.stopped_early


def test_time_budget_counts_time_from_earlier_attempts(tmp_path):
    from llm_utils import TimeBudgetCallback

    state_file = tmp_path / "train_time.json"
    state_file.write_text(json.dumps({"elapsed_seconds": 119.5, "step": 40}), encoding="utf-8")
    callback = TimeBudgetCallback(2, state_file)  # 2 minutes in total, 119.5 s already used
    callback.on_train_begin(None, None, None)
    assert callback.elapsed() >= 119.5
    import time
    time.sleep(0.6)
    control = _budget_step(callback, step=41)
    assert control.should_training_stop and control.should_save and control.should_evaluate
    assert json.loads(state_file.read_text(encoding="utf-8"))["elapsed_seconds"] >= 120
    # Without the file a fresh run would have had the full budget.
    fresh = TimeBudgetCallback(2)
    fresh.on_train_begin(None, None, None)
    assert not _budget_step(fresh).should_training_stop
