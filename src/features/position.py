from typing import List


def position_scores(sentences: List[str]) -> List[float]:
    n = len(sentences)
    if n == 0:
        return []
    # Higher score for earlier sentences (descending linear)
    return [1.0 - (i / max(1, n - 1)) if n > 1 else 1.0 for i in range(n)]

