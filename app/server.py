"""CodeGuard demo: a small web app that suggests a fix for a buggy Python fragment.

    python app/server.py [--host 127.0.0.1] [--port 8000]      then open http://127.0.0.1:8000
    uvicorn app.server:app                                     (from the repository root)

Environment variables (read once, when the server starts):
    CODEGUARD_MODEL      model id or local directory (default deepseek-ai/deepseek-coder-1.3b-instruct)
    CODEGUARD_REVISION   optional model revision (commit hash)
    CODEGUARD_ADAPTER    LoRA adapter directory; "none" or empty for the base model. If unset, the
                         small-tier 1.3B adapter is used when it exists and the model is the default.
    CODEGUARD_QUANTIZED  "1" to load the model in 4-bit (needs a CUDA GPU)
    CODEGUARD_DTYPE      auto (default), bf16, fp16 or fp32; bf16 halves memory on CPU (auto is fp32 there)

The model is loaded once at startup, on the GPU if one is visible, else on the CPU (slow). To force
the CPU, hide the GPU: CUDA_VISIBLE_DEVICES=-1. Requests are served one at a time.
"""

import argparse
import difflib
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from data_utils import MAX_SNIPPET_CHARS, MAX_SNIPPET_LINES, normalize_code  # noqa: E402
from llm_utils import load_model, model_device  # noqa: E402
from repair_utils import edited_lines, generate_repairs, parse_output  # noqa: E402

DEFAULT_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
DEFAULT_ADAPTER = ROOT / "runs" / "lora_1.3b_r16_small" / "adapter"
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

log = logging.getLogger("uvicorn.error")  # uvicorn configures this logger, so messages are shown
state = {}
generation_lock = threading.Lock()


def choose_adapter(model_id):
    value = os.environ.get("CODEGUARD_ADAPTER")
    if value is None:
        # The default adapter was trained on the default model; never put it on another one.
        if model_id == DEFAULT_MODEL and DEFAULT_ADAPTER.is_dir():
            return str(DEFAULT_ADAPTER)
        return None
    if value.strip().lower() in ("", "none"):
        return None
    return value


def load():
    model_id = os.environ.get("CODEGUARD_MODEL") or DEFAULT_MODEL
    revision = os.environ.get("CODEGUARD_REVISION") or None
    quantized = os.environ.get("CODEGUARD_QUANTIZED") == "1"
    dtype = os.environ.get("CODEGUARD_DTYPE") or "auto"
    adapter = choose_adapter(model_id)
    log.info("Loading model %s (revision %s, quantized %s, dtype %s)", model_id, revision, quantized, dtype)
    model, tokenizer = load_model(model_id, quantized=quantized, revision=revision, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    log.info("Adapter: %s", adapter or "none (base model)")
    log.info("Device: %s", model_device(model))
    return {"model": model, "tokenizer": tokenizer, "model_id": model_id, "revision": revision,
            "adapter": adapter, "quantized": quantized, "device": str(model_device(model))}


@asynccontextmanager
async def lifespan(_app):
    state.update(load())
    yield
    state.clear()


app = FastAPI(title="CodeGuard demo", lifespan=lifespan)


class RepairRequest(BaseModel):
    code: str


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    return {key: state.get(key) for key in ("model_id", "revision", "adapter", "quantized", "device")}


@app.post("/repair")
def repair(request: RepairRequest):
    code = request.code.replace("\r\n", "\n")
    if not code.strip():
        raise HTTPException(422, "The code is empty.")
    n_lines = code.count("\n") + 1
    if len(code) > MAX_SNIPPET_CHARS or n_lines > MAX_SNIPPET_LINES:
        raise HTTPException(413, f"The code is too long ({len(code)} characters, {n_lines} lines). "
                                 f"The limit is {MAX_SNIPPET_CHARS} characters and {MAX_SNIPPET_LINES} lines.")
    with generation_lock:
        start = time.perf_counter()
        [(raw, _)] = generate_repairs(state["model"], state["tokenizer"], [code], batch_size=1)
        seconds = time.perf_counter() - start
    fixed, parse_ok = parse_output(raw, prefilled=True)
    diff = difflib.unified_diff(code.split("\n"), fixed.split("\n"), "input", "suggested fix", lineterm="")
    last = normalize_code(code).count("\n")  # index of the last normalized line
    return {
        "fixed": fixed,
        # 1-based numbers of the input lines the model changed, counted AFTER normalize_code: blank
        # lines are not counted (index.html maps them back to the lines of the text box). Code added
        # after the last line is reported on the last line.
        "changed_lines": sorted({min(i, last) + 1 for i in edited_lines(code, fixed)}),
        "diff": "\n".join(diff),
        "parse_ok": parse_ok,
        "seconds": round(seconds, 2),
    }


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="CodeGuard demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
