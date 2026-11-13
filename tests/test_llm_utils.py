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
