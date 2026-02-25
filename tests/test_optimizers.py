"""Unit tests for optimizer modules."""

import pytest
import numpy as np

from src.models.extractive.greedy import greedy_select
from src.models.extractive.grasp import grasp_select


@pytest.fixture
def sample_data():
    sentences = [
        "Short sentence.",
        "A bit longer sentence here.",
        "The longest sentence in this test set right now.",
        "Another one.",
        "Medium length sentence."
    ]
    scores = [0.5, 0.8, 0.6, 0.7, 0.4]
    rng = np.random.RandomState(42)
    sim = rng.rand(5, 5)
    sim = (sim + sim.T) / 2
    np.fill_diagonal(sim, 1.0)
    return sentences, scores, sim


class TestGreedy:
    def test_empty(self):
        assert greedy_select([], [], None, 100) == []

    def test_basic(self, sample_data):
        sents, scores, sim = sample_data
        result = greedy_select(sents, scores, sim, 100)
        assert len(result) > 0
        assert all(0 <= i < len(sents) for i in result)

    def test_respects_budget(self, sample_data):
        sents, scores, sim = sample_data
        result = greedy_select(sents, scores, sim, 5, unit="tokens")
        total = sum(len(sents[i].split()) for i in result)
        assert total <= 5

    def test_sentence_unit(self, sample_data):
        sents, scores, sim = sample_data
        result = greedy_select(sents, scores, sim, 1000, unit="sentences", max_sentences=2)
        assert len(result) <= 2

    def test_sorted_output(self, sample_data):
        sents, scores, sim = sample_data
        result = greedy_select(sents, scores, sim, 50)
        assert result == sorted(result)


class TestGrasp:
    def test_empty(self):
        assert grasp_select([], [], None, 100) == []

    def test_basic(self, sample_data):
        sents, scores, sim = sample_data
        result = grasp_select(sents, scores, sim, 50, seed=42)
        assert len(result) > 0

    def test_deterministic(self, sample_data):
        sents, scores, sim = sample_data
        r1 = grasp_select(sents, scores, sim, 50, seed=42, iters=5)
        r2 = grasp_select(sents, scores, sim, 50, seed=42, iters=5)
        assert r1 == r2

    def test_sorted_output(self, sample_data):
        sents, scores, sim = sample_data
        result = grasp_select(sents, scores, sim, 50, seed=42)
        assert result == sorted(result)

    def test_respects_budget(self, sample_data):
        sents, scores, sim = sample_data
        result = grasp_select(sents, scores, sim, 6, seed=42, unit="tokens")
        total = sum(len(sents[i].split()) for i in result)
        assert total <= 6
