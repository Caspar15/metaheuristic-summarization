from typing import List
import numpy as np


def coverage_rerank(selected_idx: List[int], sim_mat: np.ndarray) -> List[int]:
    if not selected_idx:
        return selected_idx
    # Score each sentence by sum of similarity to all others (centrality)
    scores = {i: float(np.sum(sim_mat[i])) for i in selected_idx}
    return sorted(selected_idx, key=lambda i: scores[i], reverse=True)


def coherence_rerank_keep_order(selected_idx: List[int], sim_mat: np.ndarray) -> List[int]:
    # simple keep original order to preserve coherence
    return selected_idx

