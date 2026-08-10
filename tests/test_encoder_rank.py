"""Golden tests for the shared Sentence-BERT representation helpers."""

import numpy as np
import pytest
import torch

from src.models.extractive.encoder_rank import (
    EMBEDDING_CACHE_ENV,
    centroid_scores_from_embeddings,
    cosine_matrix_from_embeddings,
    encoder_document_embeddings,
)


def test_embedding_cache_miss_then_hit_is_exact(tmp_path, monkeypatch):
    calls = []

    def fake_embeddings(sentences, **kwargs):
        calls.append((list(sentences), dict(kwargs)))
        return torch.tensor([[0.6, 0.8], [1.0, 0.0]], dtype=torch.float32), {
            "model_name": kwargs["model_name"],
            "model_revision": kwargs["revision"],
            "pooling": "attention_mask_mean",
            "normalize_embeddings": True,
        }

    monkeypatch.setenv(EMBEDDING_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(
        "src.models.extractive.encoder_rank._sentence_embeddings", fake_embeddings
    )
    kwargs = {
        "model_name": "sentence-transformers/fake",
        "revision": "fixed-revision",
        "device": "cpu",
        "batch_size": 2,
        "max_model_tokens": 32,
    }
    first, first_meta = encoder_document_embeddings(["alpha", "beta"], **kwargs)
    second, second_meta = encoder_document_embeddings(["alpha", "beta"], **kwargs)

    assert len(calls) == 1
    assert torch.equal(first, second)
    assert first_meta["embedding_cache"]["status"] == "miss_written"
    assert second_meta["embedding_cache"]["status"] == "hit"
    assert (
        first_meta["embedding_cache"]["cache_key"]
        == second_meta["embedding_cache"]["cache_key"]
    )


def test_embedding_cache_key_changes_with_sentence_sequence(tmp_path, monkeypatch):
    calls = []

    def fake_embeddings(sentences, **kwargs):
        calls.append(list(sentences))
        return torch.ones((len(sentences), 2), dtype=torch.float32), {
            "model_name": kwargs["model_name"],
            "model_revision": kwargs["revision"],
        }

    monkeypatch.setenv(EMBEDDING_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(
        "src.models.extractive.encoder_rank._sentence_embeddings", fake_embeddings
    )
    kwargs = {"model_name": "fake", "revision": "rev", "device": "cpu"}
    _, meta_a = encoder_document_embeddings(["a", "bc"], **kwargs)
    _, meta_b = encoder_document_embeddings(["ab", "c"], **kwargs)
    assert len(calls) == 2
    assert meta_a["embedding_cache"]["cache_key"] != meta_b["embedding_cache"]["cache_key"]


def test_embedding_cache_corruption_fails_loud(tmp_path, monkeypatch):
    def fake_embeddings(sentences, **kwargs):
        return torch.ones((len(sentences), 2), dtype=torch.float32), {
            "model_name": kwargs["model_name"],
            "model_revision": kwargs["revision"],
        }

    cache_root = tmp_path / "cache"
    monkeypatch.setenv(EMBEDDING_CACHE_ENV, str(cache_root))
    monkeypatch.setattr(
        "src.models.extractive.encoder_rank._sentence_embeddings", fake_embeddings
    )
    kwargs = {"model_name": "fake", "revision": "rev", "device": "cpu"}
    _, metadata = encoder_document_embeddings(["a"], **kwargs)
    key = metadata["embedding_cache"]["cache_key"]
    path = cache_root / key[:2] / f"{key}.npz"
    path.write_bytes(b"not-an-npz")

    with pytest.raises(ValueError, match="invalid embedding cache artifact"):
        encoder_document_embeddings(["a"], **kwargs)


def test_normalized_embedding_cosine_matrix_golden():
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2 ** -0.5, 2 ** -0.5]],
        dtype=torch.float32,
    )
    matrix = cosine_matrix_from_embeddings(embeddings)
    expected = np.array(
        [
            [1.0, 0.0, 2 ** -0.5],
            [0.0, 1.0, 2 ** -0.5],
            [2 ** -0.5, 2 ** -0.5, 1.0],
        ]
    )
    assert matrix == pytest.approx(expected, abs=1e-7)


def test_centroid_scores_are_computed_after_sentence_normalization():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    scores = centroid_scores_from_embeddings(embeddings)
    assert scores == pytest.approx([2 ** -0.5, 2 ** -0.5], abs=1e-7)


def test_cosine_matrix_rejects_nonfinite_embeddings():
    with pytest.raises(ValueError, match="non-finite"):
        cosine_matrix_from_embeddings(np.array([[1.0, np.nan]]))
