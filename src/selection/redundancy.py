from typing import List
import numpy as np


def mmr_select(
    base_scores: List[float],
    sim_mat: np.ndarray,
    max_steps: int,
    alpha: float = 0.7,
    mask: List[bool] | None = None,
) -> List[int]:
    n = len(base_scores)
    selected: List[int] = []
    available = [i for i in range(n) if (mask[i] if mask is not None else True)]
    for _ in range(max_steps):
        best_i = None
        best_score = -1e9
        for i in available:
            if i in selected:
                continue
            if selected:
                max_sim = float(np.max(sim_mat[i, selected]))
            else:
                max_sim = 0.0
            score = alpha * base_scores[i] - (1 - alpha) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
    return selected

