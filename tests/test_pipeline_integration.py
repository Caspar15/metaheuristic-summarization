"""Integration tests for the full pipeline."""

import pytest

from src.pipeline.select_sentences import summarize_one


@pytest.fixture
def sample_doc():
    return {
        "id": "test_001",
        "sentences": [
            "This is the first sentence about artificial intelligence.",
            "Machine learning is a subset of AI.",
            "Deep learning has revolutionized many fields.",
            "Natural language processing is important.",
            "Computer vision has many applications.",
        ],
        "highlights": "AI and machine learning are transforming technology.",
    }


@pytest.fixture
def base_config():
    return {
        "optimizer": {"method": "greedy"},
        "length_control": {"unit": "tokens", "max_tokens": 30},
        "redundancy": {"lambda": 0.7},
        "representations": {"use": True, "method": "tfidf"},
        "candidates": {"use": False},
    }


class TestPipelineGreedy:
    def test_basic(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        assert "summary" in result
        assert "selected_indices" in result
        assert "id" in result
        assert result["id"] == "test_001"
        assert len(result["summary"]) > 0

    def test_indices_valid(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        n = len(sample_doc["sentences"])
        assert all(0 <= i < n for i in result["selected_indices"])

    def test_indices_sorted(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        assert result["selected_indices"] == sorted(result["selected_indices"])

    def test_summary_matches_indices(self, sample_doc, base_config):
        result = summarize_one(sample_doc, base_config)
        expected = " ".join(sample_doc["sentences"][i] for i in result["selected_indices"])
        assert result["summary"] == expected


class TestPipelineGrasp:
    def test_grasp(self, sample_doc, base_config):
        base_config["optimizer"]["method"] = "grasp"
        base_config["seed"] = 42
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0


class TestPipelineWithV2Features:
    def test_v2_tf_isf(self, sample_doc, base_config):
        base_config["features"] = {
            "tf_isf": {"version": "v2", "use_stopwords": True},
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0

    def test_v2_position(self, sample_doc, base_config):
        base_config["features"] = {
            "position": {"version": "v2", "method": "inverse"},
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0

    def test_v2_fusion(self, sample_doc, base_config):
        base_config["features"] = {
            "fusion": {"version": "v2"},
            "weights": {"importance": 0.8, "length": 0.2, "position": 0.3},
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0

    def test_semantic_features(self, sample_doc, base_config):
        base_config["features"] = {
            "weights": {
                "importance": 0.6,
                "length": 0.1,
                "position": 0.2,
                "centrality": 0.3,
                "novelty": 0.2,
            },
        }
        result = summarize_one(sample_doc, base_config)
        assert len(result["selected_indices"]) > 0


class TestPipelineEdgeCases:
    def test_empty_sentences(self, base_config):
        doc = {"id": "empty", "sentences": [], "highlights": ""}
        result = summarize_one(doc, base_config)
        assert result["selected_indices"] == []
        assert result["summary"] == ""

    def test_single_sentence(self, base_config):
        doc = {"id": "single", "sentences": ["Hello world."], "highlights": "Hello."}
        result = summarize_one(doc, base_config)
        assert result["selected_indices"] == [0]
