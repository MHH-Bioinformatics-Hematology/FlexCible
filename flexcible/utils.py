"""Small shared helpers: seeding, logging, VRAM reporting."""
from __future__ import annotations

import logging
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "flexcible") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def vram_summary() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda: n/a"
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.max_memory_reserved() / 1e9
        return f"vram alloc={a:.1f}GB peak_reserved={r:.1f}GB"
    except Exception:
        return "vram: n/a"
