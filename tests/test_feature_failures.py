"""Enabled feature failures must invalidate the run instead of becoming zeros."""

import numpy as np
import pytest

from src.pipeline.feature_builder import build_base_scores


SENTENCES = ["Alpha beta.", "Gamma delta."]


def test_enabled_novelty_requires_similarity_matrix():
    cfg = {"features": {"weights": {"novelty": 1.0}}}
    with pytest.raises(RuntimeError, match="no similarity matrix"):
        build_base_scores(SENTENCES, cfg, similarity_matrix=None)


def test_enabled_graph_failure_is_not_replaced_with_zeros(monkeypatch):
    cfg = {"features": {"weights": {"graph": 1.0}}}

    def fail_graph(*args, **kwargs):
        raise ValueError("broken graph")

    monkeypatch.setattr(
        "src.pipeline.feature_builder.compute_textrank_scores", fail_graph
    )
    with pytest.raises(RuntimeError, match="enabled graph feature failed"):
        build_base_scores(SENTENCES, cfg, similarity_matrix=np.eye(2))


def test_disabled_graph_does_not_require_similarity_matrix():
    cfg = {"features": {"weights": {"graph": 0.0}}}
    scores = build_base_scores(SENTENCES, cfg, similarity_matrix=None)
    assert len(scores) == len(SENTENCES)


def test_document_position_scope_requires_sentence_records():
    cfg = {
        "features": {
            "position": {"scope": "document", "version": "v2"},
            "weights": {"importance": 1.0, "position": 0.2},
        }
    }
    with pytest.raises(ValueError, match="requires one canonical sentence record"):
        build_base_scores(SENTENCES, cfg, similarity_matrix=None)
