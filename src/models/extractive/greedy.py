from typing import List, Optional
import numpy as np

from src.selection.length_controller import will_fit, trim_to_max_tokens


def greedy_select(
    sentences: List[str],
    base_scores: List[float],
    sim_mat: Optional[np.ndarray],
    max_tokens: int,
    alpha: float = 0.7,
) -> List[int]:
    selected: List[int] = []
    current: List[str] = []
    n = len(sentences)
    candidates = list(range(n))
    while True:
        best_i = None
        best_score = -1e9
        for i in candidates:
            if i in selected:
                continue
            if not will_fit(current, sentences[i], max_tokens):
                continue
            if selected and sim_mat is not None:
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
        current.append(sentences[best_i])
        # stop if no more can fit
        any_fit = any(
            (j not in selected) and will_fit(current, sentences[j], max_tokens)
            for j in candidates
        )
        if not any_fit:
            break
    # keep original order to form summary
    selected.sort()
    return selected

