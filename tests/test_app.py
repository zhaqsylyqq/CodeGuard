"""Tests of the demo server (app/server.py) on a tiny random model: no downloads, CPU is enough."""

import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # needed by fastapi.testclient

from fastapi.testclient import TestClient  # noqa: E402

from tiny_models import save_tiny_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
server = importlib.import_module("app.server")

BUGGY = "def is_adult(age):\n\n    return age > 18"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    return str(save_tiny_model(tmp_path_factory.mktemp("tiny")))


@pytest.fixture
def app_client(tiny_model, monkeypatch):
    monkeypatch.setenv("CODEGUARD_MODEL", tiny_model)
    monkeypatch.setenv("CODEGUARD_ADAPTER", "none")  # never pick up a real adapter from runs/
    monkeypatch.delenv("CODEGUARD_REVISION", raising=False)
    monkeypatch.delenv("CODEGUARD_QUANTIZED", raising=False)
    monkeypatch.delenv("CODEGUARD_DTYPE", raising=False)
    # Run on CPU: the GPU may be busy with a training run. load_model and resolve_dtype read this.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []

    def counting_load_model(*args, **kwargs):
        calls.append(args)
        return real_load_model(*args, **kwargs)

    real_load_model = server.load_model
    monkeypatch.setattr(server, "load_model", counting_load_model)
    with TestClient(server.app) as client:
        yield client, calls


def test_index_serves_page(app_client):
    client, _ = app_client
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<textarea" in response.text
    assert "may be wrong" in response.text


def test_repair_returns_all_fields(app_client):
    client, _ = app_client
    response = client.post("/repair", json={"code": BUGGY})
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"fixed", "changed_lines", "diff", "parse_ok", "seconds"}
    assert isinstance(data["fixed"], str)
    assert isinstance(data["diff"], str)
    assert isinstance(data["parse_ok"], bool)
    assert isinstance(data["seconds"], float) and data["seconds"] >= 0
    lines = data["changed_lines"]
    assert lines == sorted(set(lines))
    assert all(isinstance(n, int) and 1 <= n <= 2 for n in lines)  # 2 non-blank input lines


def test_rejects_empty_and_too_long_input(app_client):
    client, _ = app_client
    assert client.post("/repair", json={"code": "  \n "}).status_code == 422
    assert client.post("/repair", json={}).status_code == 422
    too_many_chars = "x = 1  # " + "a" * server.MAX_SNIPPET_CHARS
    too_many_lines = "\n".join(["x = 1"] * (server.MAX_SNIPPET_LINES + 1))
    for code in (too_many_chars, too_many_lines):
        response = client.post("/repair", json={"code": code})
        assert response.status_code == 413
        assert "too long" in response.json()["detail"]


def test_model_loaded_once(app_client):
    client, calls = app_client
    for _ in range(2):
        assert client.post("/repair", json={"code": BUGGY}).status_code == 200
    assert len(calls) == 1
    health = client.get("/health").json()
    assert health["adapter"] is None
    assert health["device"] == "cpu"


def test_adapter_choice(monkeypatch):
    monkeypatch.setenv("CODEGUARD_ADAPTER", "none")
    assert server.choose_adapter(server.DEFAULT_MODEL) is None
    monkeypatch.setenv("CODEGUARD_ADAPTER", "")
    assert server.choose_adapter(server.DEFAULT_MODEL) is None
    monkeypatch.setenv("CODEGUARD_ADAPTER", "some/adapter")
    assert server.choose_adapter("other/model") == "some/adapter"
    monkeypatch.delenv("CODEGUARD_ADAPTER")
    assert server.choose_adapter("other/model") is None  # the default adapter belongs to the default model
