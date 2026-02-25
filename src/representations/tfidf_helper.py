"""Unified TF-IDF utilities used across the pipeline.

Centralises TF-IDF vectorisation, cosine-similarity matrices, and
centroid-similarity scores so that no module needs to duplicate this logic.
"""

from typing import List, Tuple

import numpy as np


def tfidf_scores_and_sim(
    sentences: List[str],
    *,
    lowercase: bool = True,
    sublinear_tf: bool = True,
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str | None = None,
) -> Tuple[List[float], np.ndarray]:
    """Return (centroid_scores, pairwise_sim) in one pass.

    Parameters
    ----------
    sentences : list of str
    lowercase, sublinear_tf, ngram_range, stop_words :
        Forwarded to ``TfidfVectorizer``.

    Returns
    -------
    centroid_scores : list[float]
        Cosine similarity of each sentence to the document centroid.
    sim : ndarray, shape (N, N)
        Pairwise cosine similarity matrix.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not sentences:
        return [], np.zeros((0, 0))

    X = TfidfVectorizer(
        lowercase=lowercase,
        sublinear_tf=sublinear_tf,
        ngram_range=ngram_range,
        stop_words=stop_words,
    ).fit_transform(sentences)

    # centroid similarity
    doc = np.asarray(X.mean(axis=0))
    if doc.ndim == 1:
        doc = doc.reshape(1, -1)
    centroid_scores = cosine_similarity(X, doc).ravel().tolist()

    # pairwise similarity
    sim = cosine_similarity(X)
    return centroid_scores, sim


def tfidf_centroid_ranks(sentences: List[str], k: int) -> List[int]:
    """Return top-*k* sentence indices ranked by TF-IDF centroid similarity."""
    if not sentences:
        return []
    scores, _ = tfidf_scores_and_sim(sentences, sublinear_tf=False, ngram_range=(1, 1))
    idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    return idx[:k]
