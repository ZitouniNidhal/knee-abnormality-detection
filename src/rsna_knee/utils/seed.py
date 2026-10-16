"""Seed management for full experiment reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Set all random seeds and optionally enable deterministic CUDA ops.

    Parameters
    ----------
    seed:
        Integer seed to use across Python, NumPy, and PyTorch.
    deterministic:
        When *True*, sets ``torch.use_deterministic_algorithms(True)`` and
        ``CUBLAS_WORKSPACE_CONFIG`` so that CUDA operations are reproducible at
        the cost of some throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
