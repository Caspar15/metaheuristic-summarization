"""Golden and contract tests for clean-room PacSum adaptations."""

import numpy as np
import pytest
import torch

from src.baselines.pacsum import (
    pacsum_centrality_scores,
    summarize_one_pacsum_sbert,
    summarize_one_pacsum_tfidf,
)
from src.data.schemas import build_document_example


def _doc():
    return build_document_example(
        example_id="pacsum_doc",
        split="validation",
        documents=[[
            "Alpha topic appears with shared context words.",
            "Shared context words continue the alpha topic.",
            "A separate closing point mentions omega.",
        ]],
        references=["reference"],
        input_mode="single_document",
        output_mode="multi_sentence",
        dataset_name="toy",
    )


def _config(max_words=8):
    return {
        "length_control": {
            "unit": "words",
            "min_words": 0,
            "max_words": max_words,
            "require_nonempty": True,
        },
        "baselines": {
            "pacsum": {
                "beta": 0.0,
                "lambda_previous": 0.0,
                "lambda_following": 1.0,
                "tfidf": {"ngram_range": [1, 1]},
            }
        },
        "routes": {
            "semantic": {
                "model_name": "sentence-transformers/fake",
                "revision": "pinned-revision",
            }
        },
    }


def test_pacsum_centrality_hand_computed_directionality():
    similarities = np.array(
        [[1.0, 0.9, 0.2], [0.9, 1.0, 0.4], [0.2, 0.4, 1.0]]
    )
    scores, threshold = pacsum_centrality_scores(
        similarities,
        beta=0.0,
        lambda_previous=0.0,
        lambda_following=1.0,
    )
    assert threshold == pytest.approx(0.2)
    assert scores == pytest.approx([0.7, 0.2, 0.0])

    reverse_scores, _ = pacsum_centrality_scores(
        similarities,
        beta=0.0,
        lambda_previous=-1.0,
        lambda_following=0.0,
    )
    assert reverse_scores == pytest.approx([0.0, -0.7, -0.2])


def test_pacsum_beta_one_removes_every_nonself_edge():
    scores, threshold = pacsum_centrality_scores(
        np.array([[1.0, 0.5], [0.5, 1.0]]),
        beta=1.0,
        lambda_previous=0.0,
        lambda_following=1.0,
    )
    assert threshold == 1.0
    assert scores.tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    "patch",
    [
        {"beta": -0.1},
        {"lambda_previous": 0.1, "lambda_following": 1.1},
        {"lambda_previous": -0.2, "lambda_following": 0.7},
    ],
)
def test_pacsum_invalid_hyperparameters_fail_loud(patch):
    cfg = _config()
    cfg["baselines"]["pacsum"].update(patch)
    with pytest.raises(ValueError):
        summarize_one_pacsum_tfidf(_doc(), cfg)


def test_pacsum_tfidf_is_governed_rank_once_baseline():
    result = summarize_one_pacsum_tfidf(_doc(), _config())
    assert result["objective_spec"]["method"] == "pacsum_tfidf"
    assert result["baseline_diagnostics"]["status"] == "clean_room_protocol_adaptation"
    assert result["output_budget"]["min_words_applied"] is False
    assert result["selected_indices"]


def test_pacsum_sbert_uses_pinned_encoder_and_records_hashes(monkeypatch):
    def fake_embeddings(sentences, **kwargs):
        values = torch.tensor(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=torch.float32
        )
        return values, {
            "model_name": kwargs["model_name"],
            "model_revision": kwargs["revision"],
            "normalize_embeddings": True,
        }

    monkeypatch.setattr(
        "src.baselines.pacsum.encoder_document_embeddings", fake_embeddings
    )
    result = summarize_one_pacsum_sbert(_doc(), _config())
    diagnostics = result["baseline_diagnostics"]
    assert diagnostics["representation"]["model_revision"] == "pinned-revision"
    assert len(diagnostics["similarity_sha256"]) == 64
    assert len(diagnostics["centrality_sha256"]) == 64
