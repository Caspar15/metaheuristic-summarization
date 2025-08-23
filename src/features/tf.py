from __future__ import annotations

import math
from collections import Counter
from typing import List


def tf_score(tokens: List[str], norm: str = "none") -> float:
    """Compute a TF-based score for a sentence.

    Uses sum over token frequencies; with norm="log" uses log(1+tf).
    """
    counts = Counter(tokens)
    if norm == "log":
        return float(sum(math.log(1 + c) for c in counts.values()))
    return float(sum(counts.values()))

