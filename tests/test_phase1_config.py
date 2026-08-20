"""Static guardrails for the validation-only Phase 1 MVP configuration."""

from pathlib import Path

import pytest

from src.data.schemas import build_document_example
from src.pipeline.select_sentences import summarize_one, validate_experiment_request
from src.utils.io import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_mvp_config_keeps_budgets_and_routes_explicit():
    cfg = load_yaml(str(ROOT / "configs" / "phase1_mvp_multinews.yaml"))
    assert cfg["experiment"]["status"] == "validation_pilot_only"
    assert cfg["data_policy"] == {
        "policy_path": "configs/data_policies/multinews_validation_v1.json",
        "policy_sha256": "dd72b2c39e92511cf7e3d1d944eb86e34ae7f197d0f188f46ada85076ef3c615",
        "analysis": "main",
    }
    assert cfg["compute_budget"] == {
        "mode": "fixed",
        "enabled_routes": ["lexical", "semantic"],
    }
    assert cfg["candidate_budget"]["route_top_k"] > 0
    assert 0 < cfg["candidate_budget"]["min_per_route"] <= cfg[
        "candidate_budget"
    ]["route_top_k"]
    assert cfg["candidate_budget"]["total"] >= 2 * cfg["candidate_budget"][
        "min_per_route"
    ]
    assert cfg["selector"]["salience_source"] == "rrf_fusion"
    assert cfg["length_control"]["max_words"] > 0
    assert cfg["routes"]["semantic"]["revision"] != "main"
    assert len(cfg["routes"]["semantic"]["revision"]) == 40
    assert cfg["optimizer"]["method"] == "greedy"


def _phase1_config():
    return load_yaml(str(ROOT / "configs" / "phase1_mvp_multinews.yaml"))


def _canonical_row(*, split="validation", dataset_name="Multi-News"):
    return build_document_example(
        example_id=f"{split}-row",
        split=split,
        documents=[["One source sentence."]],
        references=["Reference."],
        input_mode="multi_document",
        output_mode="multi_sentence",
        dataset_name=dataset_name,
        metadata={"n_source_documents": 1},
    )


def test_validation_pilot_config_rejects_test_before_running():
    cfg = _phase1_config()
    with pytest.raises(ValueError, match="may only access the validation split"):
        validate_experiment_request(cfg, "test")
    with pytest.raises(ValueError, match="may only access the validation split"):
        summarize_one(_canonical_row(split="test"), cfg)


def test_validation_pilot_requires_canonical_matching_dataset():
    cfg = _phase1_config()
    with pytest.raises(ValueError, match="canonical rows"):
        summarize_one({"id": "legacy", "sentences": ["Sentence."]}, cfg)
    with pytest.raises(ValueError, match="does not match"):
        summarize_one(_canonical_row(dataset_name="Other Dataset"), cfg)


def test_unknown_experiment_status_fails_closed():
    cfg = _phase1_config()
    cfg["experiment"]["status"] = "typo"
    with pytest.raises(ValueError, match="unknown experiment.status"):
        validate_experiment_request(cfg, "validation")


def test_final_test_only_requires_exact_govreport_policy_and_full_test():
    cfg = _phase1_config()
    cfg["experiment"] = {"status": "final_test_only", "dataset": "GovReport"}
    cfg["data_policy"] = {
        "policy_path": "configs/data_policies/govreport_test_v1.json",
        "policy_sha256": "0" * 64,
        "analysis": "main",
    }
    validate_experiment_request(cfg, "test")
    with pytest.raises(ValueError, match="may only access the test split"):
        validate_experiment_request(cfg, "validation")
    cfg["experiment_partition"] = {"name": "dev"}
    with pytest.raises(ValueError, match="complete official test"):
        validate_experiment_request(cfg, "test")


def test_governed_experiment_cannot_omit_data_policy():
    cfg = _phase1_config()
    del cfg["data_policy"]
    with pytest.raises(ValueError, match="require a data_policy"):
        validate_experiment_request(cfg, "validation")


def test_selector_comparison_config_freezes_matched_sbert_inputs():
    cfg = load_yaml(str(ROOT / "configs" / "selector_comparison_multinews.yaml"))
    assert cfg["experiment"]["status"] == "validation_pilot_only"
    assert cfg["compute_budget"]["enabled_routes"] == ["lexical", "semantic"]
    assert cfg["selector"] == {
        "salience_source": "semantic_raw",
        "similarity_source": "sbert",
    }
    assert cfg["optimizer"]["method"] == "greedy"
    assert cfg["optimizer"]["lambda_relevance"] == pytest.approx(0.7)
    assert cfg["routes"]["semantic"]["revision"] == (
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    )
