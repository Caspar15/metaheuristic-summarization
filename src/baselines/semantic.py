"""Full-source Sentence-BERT centroid and MMR baselines."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping

import numpy as np

from src.baselines.contract import select_by_score, summarize_one_baseline
from src.models.extractive.encoder_rank import (
    centroid_scores_from_embeddings,
    cosine_matrix_from_embeddings,
    encoder_document_embeddings,
)
from src.models.extractive.mmr import mmr_select


CENTROID_MIN_WORDS_NOT_APPLIED_REASON = (
    "SBERT centroid-only is a rank-once/fill-once baseline; min_words is not "
    "used to alter its ranking, matching the shared rank-based baseline policy"
)


def _semantic_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    semantic = (cfg.get("routes", {}) or {}).get("semantic", {})
    model_name = semantic.get("model_name")
    revision = semantic.get("revision")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("SBERT baselines require routes.semantic.model_name")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("SBERT baselines require a pinned routes.semantic.revision")
    return {
        "model_name": model_name,
        "revision": revision,
        "device": semantic.get("device"),
        "batch_size": int(semantic.get("batch_size", 16)),
        "max_model_tokens": int(semantic.get("max_model_tokens", 256)),
    }


def _hash(values, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _encode_eligible(records, encoder_cfg, captured):
    embeddings, metadata = encoder_document_embeddings(
        [record["text"] for record in records], **encoder_cfg
    )
    scores = centroid_scores_from_embeddings(embeddings)
    similarities = cosine_matrix_from_embeddings(embeddings)
    captured.update(
        {
            "representation": metadata,
            "eligible_original_indices_sha256": _hash(
                [record["original_index"] for record in records], "<i8"
            ),
            "centroid_relevance_sha256": _hash(scores, "<f8"),
            "similarity_sha256": _hash(similarities, "<f8"),
        }
    )
    return scores, similarities


def summarize_one_sbert_centroid(
    doc: Mapping[str, Any], cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    encoder_cfg = _semantic_config(cfg)
    diagnostics: Dict[str, Any] = {
        "method": "sbert_centroid",
        "scope": "full_eligible_source",
    }

    def select_fn(records, evaluator):
        scores, _ = _encode_eligible(records, encoder_cfg, diagnostics)
        return select_by_score(records, evaluator, scores)

    result = summarize_one_baseline(
        doc,
        cfg,
        method="sbert_centroid",
        select_fn=select_fn,
        length_gate=True,
        apply_min_words=False,
        min_words_not_applied_reason=CENTROID_MIN_WORDS_NOT_APPLIED_REASON,
    )
    result["baseline_diagnostics"] = diagnostics
    return result


def summarize_one_sbert_mmr(
    doc: Mapping[str, Any], cfg: Mapping[str, Any]
) -> Dict[str, Any]:
    encoder_cfg = _semantic_config(cfg)
    baseline_cfg = (cfg.get("baselines", {}) or {}).get("sbert_mmr", {})
    lambda_relevance = float(baseline_cfg.get("lambda_relevance", 0.7))
    diagnostics: Dict[str, Any] = {
        "method": "sbert_mmr",
        "scope": "full_eligible_source",
        "lambda_relevance": lambda_relevance,
    }

    def select_fn(records, evaluator):
        scores, similarities = _encode_eligible(records, encoder_cfg, diagnostics)
        mmr_diagnostics: Dict[str, Any] = {}
        selected = mmr_select(
            [record["text"] for record in records],
            scores,
            similarities,
            evaluator.constraints.max_length or len(records),
            lambda_relevance=lambda_relevance,
            unit=evaluator.constraints.length_unit,
            max_sentences=evaluator.constraints.max_sentences,
            evaluator=evaluator,
            diagnostics=mmr_diagnostics,
            assert_feasible=False,
        )
        diagnostics["selection"] = mmr_diagnostics
        return selected

    result = summarize_one_baseline(
        doc,
        cfg,
        method="sbert_mmr",
        select_fn=select_fn,
        length_gate=True,
        apply_min_words=True,
    )
    result["baseline_diagnostics"] = diagnostics
    return result

