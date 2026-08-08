"""Clean-room PacSum-centrality baselines under the governed contract.

This module implements the directed degree-centrality equations described in
Zheng and Lapata (ACL 2019), but it is deliberately not advertised as a
bit-for-bit reproduction of the authors' released system.  Their public
repository has no LICENSE and its fine-tuned BERT checkpoint has neither a
redistribution licence nor a published digest.  We therefore provide two
auditable protocol adaptations:

``pacsum_tfidf``
    Directed PacSum centrality over an unnormalised TF-IDF dot-product matrix.

``pacsum_sbert``
    The same centrality rule over cosine similarities from this project's
    pinned sentence-transformer.

Both variants rank once and then walk that ranking under the shared hard
maximum-length contract.  Like TextRank, LexRank, and SBERT centroid, they do
not alter their ranking to satisfy ``min_words``.  Ties are resolved by frozen
canonical sentence order rather than the upstream implementation's unseeded
shuffle, making the result deterministic and independently auditable.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.baselines.contract import select_by_score, summarize_one_baseline
from src.models.extractive.encoder_rank import (
    cosine_matrix_from_embeddings,
    encoder_document_embeddings,
)


METHODS = {"pacsum_tfidf", "pacsum_sbert"}
PACSUM_MIN_WORDS_NOT_APPLIED_REASON = (
    "PacSum centrality is a rank-once/fill-once baseline; min_words is not "
    "used to alter its ranking, matching the shared rank-based baseline policy"
)


def _hash(values: Sequence[float] | np.ndarray, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _pacsum_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    raw = (cfg.get("baselines", {}) or {}).get("pacsum", {}) or {}
    beta = float(raw.get("beta", 0.0))
    lambda_previous = float(raw.get("lambda_previous", 0.0))
    lambda_following = float(raw.get("lambda_following", 1.0))
    if not 0.0 <= beta <= 1.0:
        raise ValueError("baselines.pacsum.beta must be in [0, 1]")
    if lambda_previous > 0.0:
        raise ValueError(
            "baselines.pacsum.lambda_previous must be <= 0; PacSum penalizes "
            "similarity to preceding content"
        )
    if lambda_following < 0.0:
        raise ValueError("baselines.pacsum.lambda_following must be >= 0")
    if not np.isclose(-lambda_previous + lambda_following, 1.0, atol=1e-12):
        raise ValueError(
            "PacSum tuning contract requires -lambda_previous + "
            "lambda_following == 1"
        )

    tfidf = raw.get("tfidf", {}) or {}
    ngram = tuple(int(value) for value in tfidf.get("ngram_range", [1, 1]))
    if len(ngram) != 2 or ngram[0] < 1 or ngram[1] < ngram[0]:
        raise ValueError("baselines.pacsum.tfidf.ngram_range must be [min_n, max_n]")
    return {
        "beta": beta,
        "lambda_previous": lambda_previous,
        "lambda_following": lambda_following,
        "tfidf": {
            "lowercase": bool(tfidf.get("lowercase", True)),
            "sublinear_tf": bool(tfidf.get("sublinear_tf", False)),
            "ngram_range": ngram,
            "stop_words": tfidf.get("stop_words"),
        },
    }


def pacsum_centrality_scores(
    similarities: Sequence[Sequence[float]] | np.ndarray,
    *,
    beta: float,
    lambda_previous: float,
    lambda_following: float,
) -> Tuple[np.ndarray, float]:
    """Return directed PacSum scores and the applied edge threshold.

    The threshold follows the paper's full-matrix definition, including the
    unit diagonal.  Only strictly positive, non-self, upper-triangle edges
    contribute after threshold subtraction.  ``lambda_previous`` is the
    signed coefficient from the paper (normally non-positive), whereas
    ``lambda_following`` is normally non-negative.
    """

    matrix = np.asarray(similarities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("PacSum similarities must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("PacSum similarities contain non-finite values")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("PacSum beta must be in [0, 1]")
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=float), 0.0

    threshold = float(matrix.min() + beta * (matrix.max() - matrix.min()))
    edges = np.maximum(matrix - threshold, 0.0)
    np.fill_diagonal(edges, 0.0)

    scores = np.zeros(matrix.shape[0], dtype=float)
    for index in range(matrix.shape[0]):
        previous = float(edges[index, :index].sum())
        following = float(edges[index, index + 1 :].sum())
        scores[index] = (
            lambda_previous * previous + lambda_following * following
        )
    return scores, threshold


def _tfidf_dot_similarity(sentences: Sequence[str], config: Mapping[str, Any]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        values = TfidfVectorizer(
            lowercase=config["lowercase"],
            sublinear_tf=config["sublinear_tf"],
            ngram_range=config["ngram_range"],
            stop_words=config["stop_words"],
            norm=None,
        ).fit_transform(sentences)
        matrix = np.asarray((values @ values.T).toarray(), dtype=float)
        degenerate = False
    except ValueError as error:
        if "empty vocabulary" not in str(error).lower():
            raise
        matrix = np.zeros((len(sentences), len(sentences)), dtype=float)
        degenerate = True
    np.fill_diagonal(matrix, 1.0)
    return matrix, degenerate


def _semantic_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    semantic = (cfg.get("routes", {}) or {}).get("semantic", {}) or {}
    model_name = semantic.get("model_name")
    revision = semantic.get("revision")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("pacsum_sbert requires routes.semantic.model_name")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("pacsum_sbert requires a pinned routes.semantic.revision")
    return {
        "model_name": model_name,
        "revision": revision,
        "device": semantic.get("device"),
        "batch_size": int(semantic.get("batch_size", 16)),
        "max_model_tokens": int(semantic.get("max_model_tokens", 256)),
    }


def summarize_one_pacsum(
    doc: Mapping[str, Any], cfg: Mapping[str, Any], *, method: str
) -> Dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown PacSum method {method!r}; choose one of {sorted(METHODS)}")
    params = _pacsum_config(cfg)
    diagnostics: Dict[str, Any] = {
        "method": method,
        "status": "clean_room_protocol_adaptation",
        "paper": "Zheng and Lapata, ACL 2019",
        "scope": "full_eligible_source_in_frozen_canonical_order",
        "beta": params["beta"],
        "lambda_previous": params["lambda_previous"],
        "lambda_following": params["lambda_following"],
        "tie_break": "frozen_canonical_sentence_order",
    }

    def select_fn(records, evaluator):
        sentences = [record["text"] for record in records]
        if method == "pacsum_tfidf":
            similarities, representation_degenerate = _tfidf_dot_similarity(
                sentences, params["tfidf"]
            )
            representation = {
                "kind": "unnormalized_tfidf_dot",
                **params["tfidf"],
            }
        else:
            embeddings, representation = encoder_document_embeddings(
                sentences, **_semantic_config(cfg)
            )
            similarities = cosine_matrix_from_embeddings(embeddings)
            representation_degenerate = False

        scores, threshold = pacsum_centrality_scores(
            similarities,
            beta=params["beta"],
            lambda_previous=params["lambda_previous"],
            lambda_following=params["lambda_following"],
        )
        diagnostics.update(
            {
                "representation": representation,
                "representation_degenerate": representation_degenerate,
                "score_degenerate": bool(
                    len(scores) > 1 and np.ptp(scores) <= 1e-15
                ),
                "edge_threshold": threshold,
                "similarity_sha256": _hash(similarities),
                "centrality_sha256": _hash(scores),
                "eligible_original_indices_sha256": _hash(
                    [record["original_index"] for record in records], "<i8"
                ),
            }
        )
        return select_by_score(records, evaluator, scores)

    result = summarize_one_baseline(
        doc,
        cfg,
        method=method,
        select_fn=select_fn,
        length_gate=True,
        apply_min_words=False,
        min_words_not_applied_reason=PACSUM_MIN_WORDS_NOT_APPLIED_REASON,
    )
    result["baseline_diagnostics"] = diagnostics
    return result


def summarize_one_pacsum_tfidf(
    doc: Mapping[str, Any], cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    return summarize_one_pacsum(doc, cfg, method="pacsum_tfidf")


def summarize_one_pacsum_sbert(
    doc: Mapping[str, Any], cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    return summarize_one_pacsum(doc, cfg, method="pacsum_sbert")
