from copy import deepcopy
from pathlib import Path

from scripts.audit.run_govreport_final_test import (
    FAMILY_LABELS,
    RANDOM_SEEDS,
    _baseline_config,
    _baseline_method,
)
from src.pipeline.select_sentences import validate_experiment_request
from src.utils.io import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_yaml(str(ROOT / "configs/final/govreport_final_v1.yaml"))


def test_final_config_is_complete_test_only_and_has_no_dev_partition():
    config = _config()
    validate_experiment_request(config, "test")
    assert "experiment_partition" not in config
    assert config["experiment"] == {"status": "final_test_only", "dataset": "GovReport"}
    assert config["compute_budget"]["enabled_routes"] == ["lexical", "semantic", "graph"]
    assert config["optimizer"] == {"method": "mmr", "lambda_relevance": 0.7}
    assert config["length_control"]["min_words"] == 500
    assert config["length_control"]["max_words"] == 650


def test_final_proposed_scientific_fields_equal_frozen_d3b_candidate():
    final = _config()
    source = load_yaml(
        str(
            ROOT
            / "runs_v2/d3b_cross_profile_combination_v1/govreport/dev/"
            "C01_combined_salience_route_weight/resolved_config.yaml"
        )
    )
    for key in (
        "seed",
        "features",
        "representations",
        "compute_budget",
        "candidate_budget",
        "candidates",
        "selector",
        "objectives",
        "routes",
        "coverage_guard",
        "optimizer",
        "length_control",
    ):
        assert final[key] == source[key]


def test_final_system_matrix_has_nine_families_and_ten_fixed_random_seeds():
    assert len(FAMILY_LABELS) == 9
    assert len(set(FAMILY_LABELS)) == 9
    assert FAMILY_LABELS[0] == "proposed"
    assert "sbert_mmr_lambda_0.9" in FAMILY_LABELS
    assert RANDOM_SEEDS == (3407, 2024, 42, 1337, 2026, 20260810, 7, 17, 23, 101)


def test_final_baseline_labels_resolve_without_changing_proposed_selector():
    config = _config()
    original = deepcopy(config["optimizer"])
    assert _baseline_method("random_seed_42") == "random"
    assert _baseline_method("pacsum_tfidf_P07") == "pacsum_tfidf"
    assert _baseline_method("sbert_mmr_lambda_0.9") == "sbert_mmr"
    sbert_mmr = _baseline_config(config, "sbert_mmr_lambda_0.9")
    pacsum_sbert = _baseline_config(config, "pacsum_sbert_beta_0.5")
    assert sbert_mmr["baselines"]["sbert_mmr"]["lambda_relevance"] == 0.9
    assert pacsum_sbert["baselines"]["pacsum"]["beta"] == 0.5
    assert pacsum_sbert["baselines"]["pacsum"]["lambda_previous"] == 0.0
    assert config["optimizer"] == original
