from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List

from .tf import tf_score


def _sentence_frequency(sentences: List[List[str]]) -> Dict[str, int]:
    sf: Dict[str, int] = defaultdict(int)
    for toks in sentences:
        for t in set(toks):
            sf[t] += 1
    return dict(sf)


def compute_isf(sentences: List[List[str]], smooth: float = 1.0) -> Dict[str, float]:
    """Compute ISF for tokens with N = number of sentences in a document."""
    N = max(1, len(sentences))
    sf = _sentence_frequency(sentences)
    isf: Dict[str, float] = {}
    for t, s in sf.items():
        isf[t] = math.log((N + smooth) / (s + smooth))
    return isf


def tf_isf_score(tokens: List[str], isf: Dict[str, float], norm: str = "none") -> float:
    """Compute sentence TF-ISF score compatible with tf normalization."""
    # Build per-sentence TF counts
    counts = Counter(tokens)
    score = 0.0
    for t, c in counts.items():
        tf_val = math.log(1 + c) if norm == "log" else float(c)
        score += tf_val * isf.get(t, 0.0)
    return float(score)

