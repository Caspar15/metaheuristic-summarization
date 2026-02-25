"""Unit tests for feature extraction modules."""

import pytest
import numpy as np

from src.features.tf_isf import sentence_tf_isf_scores, sentence_tf_isf_scores_v2
from src.features.position import position_scores, position_scores_v2
from src.features.length import length_scores
from src.features.compose import combine_scores, combine_scores_v2
from src.features.graph import compute_textrank_scores
from src.features.semantic import sentence_centrality_scores, sentence_novelty_scores


# ---- TF-ISF ----

class TestTfIsf:
    def test_empty(self):
        assert sentence_tf_isf_scores([]) == []

    def test_single(self):
        scores = sentence_tf_isf_scores(["hello world"])
        assert len(scores) == 1

    def test_normalized_range(self):
        sents = ["The cat sat.", "Dogs run fast.", "AI is great."]
        scores = sentence_tf_isf_scores(sents)
        assert all(0 <= s <= 1 for s in scores)

    def test_v2_empty(self):
        assert sentence_tf_isf_scores_v2([]) == []

    def test_v2_stopword_effect(self):
        sents = ["The the the cat.", "Artificial intelligence rocks."]
        v1 = sentence_tf_isf_scores(sents)
        v2 = sentence_tf_isf_scores_v2(sents, use_stopwords=True)
        # v2 should produce different scores due to stopword filtering
        assert len(v2) == 2
        assert all(0 <= s <= 1 for s in v2)

    def test_v2_sublinear_tf(self):
        sents = ["word word word word.", "other sentence here."]
        v2_sub = sentence_tf_isf_scores_v2(sents, use_sublinear_tf=True)
        v2_raw = sentence_tf_isf_scores_v2(sents, use_sublinear_tf=False)
        assert len(v2_sub) == 2
        # They can differ due to sublinear weighting
        assert v2_sub != v2_raw or all(s == 0.5 for s in v2_sub)


# ---- Position ----

class TestPosition:
    def test_empty(self):
        assert position_scores([]) == []

    def test_descending(self):
        scores = position_scores(["a", "b", "c", "d"])
        assert scores[0] >= scores[1] >= scores[2] >= scores[3]

    def test_first_is_one(self):
        scores = position_scores(["a", "b", "c"])
        assert scores[0] == 1.0

    def test_v2_inverse(self):
        scores = position_scores_v2(["a", "b", "c", "d"], method="inverse")
        assert scores[0] >= scores[1] >= scores[2] >= scores[3]
        assert scores[0] == 1.0

    def test_v2_exponential(self):
        scores = position_scores_v2(["a", "b", "c", "d"], method="exponential", decay=0.5)
        assert scores[0] >= scores[1] >= scores[2] >= scores[3]

    def test_v2_linear_matches_v1(self):
        sents = ["a", "b", "c"]
        v1 = position_scores(sents)
        v2 = position_scores_v2(sents, method="linear")
        assert v1 == pytest.approx(v2, abs=1e-9)


# ---- Length ----

class TestLength:
    def test_empty(self):
        assert length_scores([]) == []

    def test_longer_scores_higher(self):
        sents = ["a", "a b c d e f g"]
        scores = length_scores(sents)
        assert scores[1] > scores[0]


# ---- Compose ----

class TestCompose:
    def test_empty(self):
        assert combine_scores({}, {}) == []

    def test_single_feature(self):
        feats = {"a": [0.5, 1.0, 0.3]}
        w = {"a": 1.0}
        result = combine_scores(feats, w)
        assert len(result) == 3
        assert max(result) == pytest.approx(1.0)

    def test_v2_minmax(self):
        feats = {"a": [10.0, 20.0, 30.0]}
        w = {"a": 1.0}
        result = combine_scores_v2(feats, w)
        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_v2_interactions(self):
        feats = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        w = {"a": 1.0, "b": 1.0}
        r_no_int = combine_scores_v2(feats, w)
        r_with_int = combine_scores_v2(feats, w, interactions=[("a", "b", 2.0)])
        assert len(r_with_int) == 2


# ---- Graph ----

class TestGraph:
    def test_empty(self):
        sim = np.zeros((0, 0))
        assert compute_textrank_scores(sim) == []

    def test_single(self):
        sim = np.array([[1.0]])
        scores = compute_textrank_scores(sim)
        assert len(scores) == 1
        assert scores[0] == pytest.approx(1.0)

    def test_sum_close_to_one(self):
        rng = np.random.RandomState(42)
        sim = rng.rand(5, 5)
        sim = (sim + sim.T) / 2
        np.fill_diagonal(sim, 1.0)
        scores = compute_textrank_scores(sim)
        assert len(scores) == 5
        assert sum(scores) == pytest.approx(1.0, abs=0.05)


# ---- Semantic ----

class TestSemantic:
    def test_centrality_empty(self):
        assert sentence_centrality_scores([]) == []

    def test_centrality_basic(self):
        sents = ["cat dog", "cat", "fish bird"]
        scores = sentence_centrality_scores(sents)
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_novelty_with_sim_matrix(self):
        sim = np.array([
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])
        scores = sentence_novelty_scores(sim)
        assert len(scores) == 3
        # sentence 2 (index 2) is most novel (least similar to others)
        assert scores[2] > scores[0]
