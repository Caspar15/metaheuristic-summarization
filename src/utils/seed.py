from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic: Optional[bool] = None) -> None:
    """Set random seeds for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    # deterministic reserved for frameworks; kept for API consistency

