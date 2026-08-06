"""Golden tests for the shared Sentence-BERT representation helpers."""

import numpy as np
import pytest
import torch

from src.models.extractive.encoder_rank import (
    centroid_scores_from_embeddings,
    cosine_matrix_from_embeddings,
)


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

