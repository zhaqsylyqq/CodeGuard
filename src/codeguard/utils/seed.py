"""Global seeding for reproducibility."""

from __future__ import annotations

import importlib.util
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and (if installed) torch, including CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses only
    random.seed(seed)
    np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
