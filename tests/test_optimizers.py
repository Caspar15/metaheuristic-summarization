"""Unit tests for optimizer modules."""

import pytest
import numpy as np

from src.models.extractive.greedy import greedy_select
from src.models.extractive.grasp import grasp_select
from src.models.extractive.mmr import mmr_select
from src.objectives.evaluator import (
    ObjectiveWeights,
    SelectionConstraints,
    SelectionObjective,
)


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

    @staticmethod
    def _reference_select(evaluator):
        """The pre-optimization loop, retained only as a regression oracle."""
        selected = []
        remaining = set(range(len(evaluator.sentences)))
        while remaining:
            current = evaluator.evaluate(selected)
            ranked = []
            for candidate in sorted(remaining):
                if evaluator.can_add(selected, candidate):
                    ranked.append(
                        (
                            evaluator.evaluate(
                                selected + [candidate]
                            ).scalar_utility,
                            candidate,
                        )
                    )
            if not ranked:
                break
            value, candidate = max(ranked, key=lambda item: (item[0], -item[1]))
            if (
                selected
                and current.feasible
                and value <= current.scalar_utility + 1e-12
            ):
                break
            selected.append(candidate)
            remaining.remove(candidate)
        evaluator.assert_feasible(selected)
        return sorted(selected)

    @pytest.mark.parametrize("importance_aggregation", ["mean", "length_normalized"])
    def test_batched_search_matches_pre_optimization_loop(
        self, importance_aggregation
    ):
        for seed in range(20):
            rng = np.random.RandomState(seed)
            n = 12
            sentences = ["word " * int(rng.randint(1, 6)) for _ in range(n)]
            similarities = rng.uniform(-0.2, 1.0, size=(n, n))
            similarities = (similarities + similarities.T) / 2.0
            np.fill_diagonal(similarities, 1.0)
            coverage = rng.uniform(-0.2, 1.0, size=(17, n))
            evaluator = SelectionObjective(
                sentences,
                rng.uniform(0.0, 1.0, size=n),
                similarities,
                coverage_matrix=coverage,
                importance_aggregation=importance_aggregation,
                coverage_method="max",
                weights=ObjectiveWeights(1.0, 0.8, 0.7),
                constraints=SelectionConstraints(
                    length_unit="words",
                    min_words=8,
                    max_length=20,
                    require_nonempty=True,
                ),
            )
            expected = self._reference_select(evaluator)
            actual = greedy_select(
                evaluator.sentences,
                evaluator.importance.tolist(),
                evaluator.similarity_matrix,
                20,
                evaluator=evaluator,
            )
            assert actual == expected


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


class TestMMR:
    @staticmethod
    def _evaluator(sentences, similarities, *, max_words=100, max_sentences=None):
        return SelectionObjective(
            sentences,
            [0.0] * len(sentences),
            similarities,
            weights=ObjectiveWeights(0.0, 0.0, 0.0),
            constraints=SelectionConstraints(
                length_unit="words",
                max_length=max_words,
                max_sentences=max_sentences,
                min_words=0,
                require_nonempty=True,
            ),
        )

    def test_golden_redundancy_changes_second_choice(self):
        sentences = ["alpha", "alpha copy", "different topic"]
        relevance = [1.0, 0.9, 0.8]
        similarities = np.array(
            [[1.0, 0.99, 0.0], [0.99, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        diagnostics = {}
        selected = mmr_select(
            sentences,
            relevance,
            similarities,
            100,
            lambda_relevance=0.5,
            evaluator=self._evaluator(
                sentences, similarities, max_sentences=2
            ),
            diagnostics=diagnostics,
        )
        assert selected == [0, 2]
        assert diagnostics["selection_order"] == [0, 2]
        assert diagnostics["steps"][1]["mmr_score"] == pytest.approx(0.4)

    def test_tie_breaks_by_lower_candidate_index(self):
        sentences = ["one", "two"]
        similarities = np.eye(2)
        diagnostics = {}
        mmr_select(
            sentences,
            [0.5, 0.5],
            similarities,
            100,
            evaluator=self._evaluator(
                sentences, similarities, max_sentences=1
            ),
            diagnostics=diagnostics,
        )
        assert diagnostics["selection_order"] == [0]

    def test_skips_candidate_that_exceeds_budget(self):
        sentences = ["one two three four", "short", "also short"]
        similarities = np.eye(3)
        selected = mmr_select(
            sentences,
            [1.0, 0.8, 0.7],
            similarities,
            3,
            evaluator=self._evaluator(sentences, similarities, max_words=3),
        )
        assert selected == [1, 2]

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_rejects_invalid_lambda(self, value):
        similarities = np.eye(1)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            mmr_select(
                ["one"],
                [1.0],
                similarities,
                1,
                lambda_relevance=value,
                evaluator=self._evaluator(["one"], similarities),
            )
