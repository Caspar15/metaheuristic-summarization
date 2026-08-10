"""Contract tests for full-source SBERT baselines."""

import torch

from src.baselines.semantic import (
    summarize_one_sbert_centroid,
    summarize_one_sbert_mmr,
)
from src.data.schemas import build_document_example


REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def _doc():
    return build_document_example(
        example_id="semantic-baseline",
        split="validation",
        documents=[["Alpha news", "Alpha update", "Sports result"]],
        references=["Reference."],
        input_mode="multi_document",
        output_mode="multi_sentence",
        dataset_name="toy",
        metadata={"n_source_documents": 1},
    )


def _config(max_words):
    return {
        "routes": {
            "semantic": {
                "model_name": "sentence-transformers/fake",
                "revision": REVISION,
                "batch_size": 3,
                "max_model_tokens": 256,
            }
        },
        "baselines": {"sbert_mmr": {"lambda_relevance": 0.5}},
        "length_control": {
            "unit": "words",
            "max_words": max_words,
            "min_words": 0,
            "require_nonempty": True,
        },
    }


def _fake_embeddings(sentences, **kwargs):
    del sentences, kwargs
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=torch.float32
    )
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings, {
        "model_name": "sentence-transformers/fake",
        "model_revision": REVISION,
        "pooling": "attention_mask_mean",
        "normalize_embeddings": True,
        "similarity": "normalized_dot_product_cosine",
        "estimated_cost": {"encoded_sentences": 3},
    }


def test_sbert_centroid_is_full_source_rank_then_fill(monkeypatch):
    monkeypatch.setattr(
        "src.baselines.semantic.encoder_document_embeddings", _fake_embeddings
    )
    result = summarize_one_sbert_centroid(_doc(), _config(max_words=2))
    assert result["selected_indices"] == [1]
    assert result["objective_spec"]["method"] == "sbert_centroid"
    assert result["baseline_diagnostics"]["scope"] == "full_eligible_source"
    assert result["baseline_diagnostics"]["similarity_sha256"]


def test_sbert_mmr_avoids_near_duplicate(monkeypatch):
    monkeypatch.setattr(
        "src.baselines.semantic.encoder_document_embeddings", _fake_embeddings
    )
    result = summarize_one_sbert_mmr(_doc(), _config(max_words=4))
    assert result["selected_indices"] == [1, 2]
    diagnostics = result["baseline_diagnostics"]
    assert diagnostics["lambda_relevance"] == 0.5
    assert diagnostics["selection"]["selection_order"] == [1, 2]

