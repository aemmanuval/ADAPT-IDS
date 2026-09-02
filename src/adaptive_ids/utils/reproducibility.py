"""Reproducibility helpers — central seed management."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set seeds for all random number generators used in the project."""
    random.seed(seed)
    np.random.seed(seed)
