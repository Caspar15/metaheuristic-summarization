"""Feature construction logic extracted from select_sentences.py."""

from typing import Dict, List, Optional

import numpy as np

from src.features.tf_isf import sentence_tf_isf_scores, sentence_tf_isf_scores_v2
from src.features.length import length_scores
from src.features.position import position_scores, position_scores_v2
from src.features.compose import combine_scores, combine_scores_v2
from src.features.graph import compute_textrank_scores
from src.features.semantic import sentence_centrality_scores, sentence_novelty_scores


def build_base_scores(
    sentences: List[str],
    cfg: Dict,
    similarity_matrix: Optional[np.ndarray] = None,
) -> List[float]:
    """Compute and fuse all base feature scores for a document.

    Reads feature configuration from *cfg* and dispatches to v1 or v2
    variants depending on ``features.tf_isf.version``,
    ``features.position.version``, and ``features.fusion.version``.
    """
    feat_cfg = cfg.get("features", {}) or {}

    # --- importance ---
    tf_isf_cfg = feat_cfg.get("tf_isf", {}) or {}
    if tf_isf_cfg.get("version", "v1") == "v2":
        f_importance = sentence_tf_isf_scores_v2(
            sentences,
            use_stopwords=bool(tf_isf_cfg.get("use_stopwords", True)),
            use_sublinear_tf=bool(tf_isf_cfg.get("use_sublinear_tf", True)),
            use_bigrams=bool(tf_isf_cfg.get("use_bigrams", False)),
        )
    else:
        f_importance = sentence_tf_isf_scores(sentences)

    # --- length ---
    f_len = length_scores(sentences)

    # --- position ---
    pos_cfg = feat_cfg.get("position", {}) or {}
    if pos_cfg.get("version", "v1") == "v2":
        f_pos = position_scores_v2(
            sentences,
            method=str(pos_cfg.get("method", "inverse")),
            decay=float(pos_cfg.get("decay", 0.1)),
        )
    else:
        f_pos = position_scores(sentences)

    # --- weights ---
    weights_cfg = feat_cfg.get("weights", {}) or {}
    weights = {
        "importance": float(weights_cfg.get("importance", cfg.get("objectives", {}).get("lambda_importance", 1.0))),
        "length": float(weights_cfg.get("length", 0.3)),
        "position": float(weights_cfg.get("position", 0.3)),
        "graph": float(weights_cfg.get("graph", 0.0)),
        "centrality": float(weights_cfg.get("centrality", 0.0)),
        "novelty": float(weights_cfg.get("novelty", 0.0)),
    }

    # --- semantic: centrality ---
    f_centrality = [0.0] * len(sentences)
    if weights["centrality"] > 1e-9:
        try:
            f_centrality = sentence_centrality_scores(sentences, similarity_matrix=similarity_matrix)
        except Exception:
            f_centrality = [0.0] * len(sentences)

    # --- semantic: novelty ---
    f_novelty = [0.0] * len(sentences)
    if weights["novelty"] > 1e-9 and similarity_matrix is not None:
        try:
            f_novelty = sentence_novelty_scores(similarity_matrix)
        except Exception:
            f_novelty = [0.0] * len(sentences)

    # --- graph ---
    f_graph: List[float] = []
    if weights["graph"] > 1e-9 and similarity_matrix is not None:
        try:
            f_graph = compute_textrank_scores(similarity_matrix)
        except Exception as e:
            print(f"Warning: Graph score computation failed: {e}")
            f_graph = [0.0] * len(sentences)
    else:
        f_graph = [0.0] * len(sentences)

    # --- fuse ---
    feats = {
        "importance": f_importance,
        "length": f_len,
        "position": f_pos,
        "graph": f_graph,
        "centrality": f_centrality,
        "novelty": f_novelty,
    }

    fusion_cfg = feat_cfg.get("fusion", {}) or {}
    if fusion_cfg.get("version", "v1") == "v2":
        raw_interactions = fusion_cfg.get("interactions", []) or []
        interactions = [(str(a), str(b), float(w)) for a, b, w in raw_interactions]
        return combine_scores_v2(feats, weights, interactions=interactions or None, normalize="minmax")
    return combine_scores(feats, weights)
