from typing import List, Optional

import numpy as np

from src.representations.tfidf_helper import tfidf_scores_and_sim


def _minmax_norm(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo = min(xs)
    hi = max(xs)
    if hi - lo < 1e-12:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _tfidf_scores_and_sim(sentences: List[str]) -> tuple[list[float], np.ndarray]:
    return tfidf_scores_and_sim(sentences)


def fast_semantic_scores_and_sim(sentences: List[str]) -> tuple[list[float], np.ndarray]:
    """Public helper: TF-IDF semantic scores to doc centroid and pairwise cosine sim."""
    return tfidf_scores_and_sim(sentences)


def fast_fused_select(
    sentences: List[str],
    base_scores: List[float],
    max_tokens: int,
    w_base: float = 0.5,
    w_sem: float = 0.5,
    alpha: float = 0.7,
    unit: str = "sentences",
    max_sentences: Optional[int] = 3,
) -> List[int]:
    from src.models.extractive.greedy import greedy_select

    if not sentences:
        return []
    sem_scores, sim = _tfidf_scores_and_sim(sentences)
    base_n = _minmax_norm(list(base_scores))
    sem_n = _minmax_norm(list(sem_scores))
    fused = [float(w_base) * base_n[i] + float(w_sem) * sem_n[i] for i in range(len(sentences))]
    picked = greedy_select(
        sentences,
        fused,
        sim,
        max_tokens,
        alpha=float(alpha),
        unit=unit,
        max_sentences=max_sentences,
    )
    return picked


def fast_grasp_select(
    sentences: List[str],
    base_scores: List[float],
    max_tokens: int,
    w_base: float = 0.5,
    w_sem: float = 0.5,
    alpha: float = 0.7,
    unit: str = "sentences",
    max_sentences: Optional[int] = 3,
    iters: int = 15,
    rcl_ratio: float = 0.3,
    seed: Optional[int] = None,
) -> List[int]:
    from src.models.extractive.grasp import grasp_select

    if not sentences:
        return []
    sem_scores, sim = _tfidf_scores_and_sim(sentences)
    base_n = _minmax_norm(list(base_scores))
    sem_n = _minmax_norm(list(sem_scores))
    fused = [float(w_base) * base_n[i] + float(w_sem) * sem_n[i] for i in range(len(sentences))]
    picked = grasp_select(
        sentences,
        fused,
        sim,
        max_tokens,
        alpha=float(alpha),
        iters=int(iters),
        rcl_ratio=float(rcl_ratio),
        seed=seed,
        unit=unit,
        max_sentences=max_sentences,
    )
    return picked


def fast_nsga2_select(
    sentences: List[str],
    base_scores: List[float],
    max_tokens: int,
    w_base: float = 0.5,
    w_sem: float = 0.5,
    unit: str = "sentences",
    max_sentences: Optional[int] = 3,
    lambda_importance: float = 1.0,
    lambda_coverage: float = 0.8,
    lambda_redundancy: float = 0.7,
    **kwargs,
) -> List[int]:
    try:
        from src.models.extractive.nsga2 import nsga2_select
    except Exception as e:
        raise RuntimeError("NSGA-II requires pymoo to be installed.") from e

    if not sentences:
        return []
    sem_scores, sim = _tfidf_scores_and_sim(sentences)
    base_n = _minmax_norm(list(base_scores))
    sem_n = _minmax_norm(list(sem_scores))
    importance = [float(w_base) * base_n[i] + float(w_sem) * sem_n[i] for i in range(len(sentences))]
    picked = nsga2_select(
        sentences,
        importance,
        sim,
        max_tokens,
        lambda_importance=float(lambda_importance),
        lambda_coverage=float(lambda_coverage),
        lambda_redundancy=float(lambda_redundancy),
        unit=unit,
        max_sentences=max_sentences,
        **kwargs,
    )
    return picked
