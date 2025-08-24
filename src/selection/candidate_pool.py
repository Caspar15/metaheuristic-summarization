from typing import List


def topk_by_score(scores: List[float], k: int) -> List[int]:
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return idx[:k]

