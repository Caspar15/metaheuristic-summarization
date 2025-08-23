from __future__ import annotations

from typing import List


def position_scores(n_sentences: int, lead_bonus: float = 0.2) -> List[float]:
    """Return a list of position scores; first sentence gets lead_bonus, others 0."""
    if n_sentences <= 0:
        return []
    scores = [0.0] * n_sentences
    scores[0] = float(lead_bonus)
    return scores

