"""Semantic features for extractive summarization.

Provides lightweight (TF-IDF-based) semantic signals that complement
the statistical features (TF-ISF, length, position).
"""

from typing import List, Optional

import numpy as np


def sentence_centrality_scores(
    sentences: List[str],
    similarity_matrix: Optional[np.ndarray] = None,
) -> List[float]:
    """Cosine similarity of each sentence to the TF-IDF document centroid.

    If *similarity_matrix* is already available the mean row is used as a
    proxy; otherwise a fresh TF-IDF vectorisation is performed.
    """
    if not sentences:
        return []

    if similarity_matrix is not None:
        # mean similarity to all other sentences ≈ centrality
        raw = similarity_matrix.mean(axis=1).tolist()
    else:
        from src.representations.tfidf_helper import tfidf_scores_and_sim
        raw, _ = tfidf_scores_and_sim(sentences, sublinear_tf=False, ngram_range=(1, 1))

    # min-max normalize
    lo = min(raw)
    hi = max(raw)
    if hi - lo > 1e-9:
        return [(v - lo) / (hi - lo) for v in raw]
    return [0.5] * len(raw)


def sentence_novelty_scores(
    similarity_matrix: np.ndarray,
) -> List[float]:
    """Novelty = 1 - average similarity to all other sentences.

    Encourages selection of diverse, non-redundant sentences.
    """
    n = similarity_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # average similarity excluding self (diagonal = 1)
    avg_sim = (similarity_matrix.sum(axis=1) - 1.0) / max(1, n - 1)
    novelty = (1.0 - avg_sim).tolist()

    lo = min(novelty)
    hi = max(novelty)
    if hi - lo > 1e-9:
        return [(v - lo) / (hi - lo) for v in novelty]
    return [0.5] * len(novelty)
