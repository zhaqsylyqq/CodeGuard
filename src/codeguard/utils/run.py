"""Run directories: every experiment gets ``runs/<timestamp>_<name>/`` with its
resolved config, environment/git metadata, log file, metrics and artefacts."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from codeguard.config import dump_config

LOGGER_NAME = "codeguard"
TRACKED_PACKAGES = (
    "codeguard",
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "accelerate",
)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Log to stderr and, optionally, to ``log_file`` (replacing previous handlers)."""
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def git_info() -> dict[str, Any]:
    """Current commit hash and whether the working tree has uncommitted changes."""

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def collect_env() -> dict[str, Any]:
    """Python/OS/hardware info and versions of the packages that affect results."""
    packages: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "gpus": [],
    }
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return env


@dataclass
class Run:
    """Handle to an active run directory."""

    id: str
    dir: Path
    started_at: str
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def path(self, *parts: str) -> Path:
        """Path inside the run directory (parent folders are created)."""
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return p

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0


@contextmanager
def start_run(
    config: BaseModel,
    name: str,
    runs_dir: Path,
    extra_meta: dict[str, Any] | None = None,
) -> Iterator[Run]:
    """Create a run directory, save config + metadata, log to ``run.log``.

    ``meta.json`` is rewritten on exit with the final status and duration, also
    when the run fails, so crashed runs are visible as ``"status": "failed"``.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{name}"
    run_dir = runs_dir / run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = runs_dir / f"{run_id}-{suffix}"
    run_dir.mkdir(parents=True)

    run = Run(id=run_dir.name, dir=run_dir, started_at=datetime.now().isoformat(timespec="seconds"))
    dump_config(config, run.path("config.yaml"))
    meta: dict[str, Any] = {
        "run_id": run.id,
        "status": "running",
        "started_at": run.started_at,
        "command": " ".join(sys.argv),
        "git": git_info(),
        "env": collect_env(),
        **(extra_meta or {}),
    }
    run.save_json("meta.json", meta)
    logger = setup_logging(run.path("run.log"))
    logger.info("Run %s started -> %s", run.id, run.dir)

    try:
        yield run
        meta["status"] = "completed"
    except BaseException:
        meta["status"] = "failed"
        logger.exception("Run %s failed", run.id)
        raise
    finally:
        meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        meta["duration_seconds"] = round(run.elapsed, 2)
        run.save_json("meta.json", meta)
        logger.info("Run %s %s in %.1fs", run.id, meta["status"], run.elapsed)
