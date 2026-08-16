from copy import deepcopy
import json
from pathlib import Path

from scripts.audit.run_multinews_final_test import (
    DETERMINISTIC_BASELINES,
    FAMILY_LABELS,
    PRIMARY_COMPARATOR,
    RANDOM_SEEDS,
    SCIENTIFIC_FIELDS,
    _baseline_config,
    _baseline_method,
    _load_freeze,
    _scientific_view,
)
from src.data.policy import sha256_file
from src.pipeline.select_sentences import validate_experiment_request
from src.utils.io import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_yaml(str(ROOT / "configs/final/multinews_final_v1.yaml"))


def test_multinews_final_config_is_test_only_and_matches_d3b_science():
    final = _config()
    source = load_yaml(
        str(
            ROOT
            / "runs_v2/d3b_cross_profile_combination_v1/multinews/dev/"
            "C01_combined_salience_route_weight/resolved_config.yaml"
        )
    )
    validate_experiment_request(final, "test")
    assert "experiment_partition" not in final
    assert final["experiment"] == {"status": "final_test_only", "dataset": "Multi-News"}
    assert tuple(_scientific_view(final)) == SCIENTIFIC_FIELDS
    assert _scientific_view(final) == _scientific_view(source)


def test_multinews_final_matrix_is_fixed_before_scores():
    assert len(FAMILY_LABELS) == 9
    assert len(set(FAMILY_LABELS)) == 9
    assert len(DETERMINISTIC_BASELINES) == 7
    assert len(RANDOM_SEEDS) == 10
    assert PRIMARY_COMPARATOR == "pacsum_tfidf_P08"
    assert "sbert_mmr_lambda_0.7" in FAMILY_LABELS
    assert "pacsum_sbert_P03" in FAMILY_LABELS


def test_multinews_dev_winner_parameters_are_resolved_without_mutation():
    config = _config()
    original = deepcopy(config)
    assert _baseline_method("random_seed_42") == "random"
    assert _baseline_method("pacsum_tfidf_P08") == "pacsum_tfidf"
    assert _baseline_method("pacsum_sbert_P03") == "pacsum_sbert"
    assert _baseline_method("sbert_mmr_lambda_0.7") == "sbert_mmr"
    assert _baseline_config(config, "pacsum_tfidf_P08")["baselines"]["pacsum"] == {
        "beta": 0.0,
        "lambda_previous": -0.8,
        "lambda_following": 0.2,
        "tfidf": config["baselines"]["pacsum"]["tfidf"],
    }
    p03 = _baseline_config(config, "pacsum_sbert_P03")
    assert p03["baselines"]["pacsum"]["lambda_previous"] == -0.3
    assert p03["baselines"]["pacsum"]["lambda_following"] == 0.7
    assert _baseline_config(config, "sbert_mmr_lambda_0.7")["baselines"]["sbert_mmr"]["lambda_relevance"] == 0.7
    assert config == original


def test_multinews_test_policy_was_frozen_before_revised_scores():
    policy = json.loads(
        (ROOT / "configs/data_policies/multinews_test_v1.json").read_text(encoding="utf-8")
    )
    assert policy["status"] == "frozen_before_revised_pipeline_test_predictions_or_scores"
    assert policy["dataset"]["official_rows"] == 5622
    assert policy["canonical_exclusions"]["source_row_indices"] == [4736]
    assert policy["analyses"]["main"]["expected_rows"] == 5621
    assert policy["analyses"]["main"]["expected_replacement_rows"] == 70
    assert policy["final_use_rule"].endswith("Never tune after test execution.")


def test_multinews_execution_freeze_is_score_blind_and_fully_pinned():
    relative = Path(
        "configs/preregistrations/multinews_final_execution_freeze_v1.json"
    )
    freeze = _load_freeze(relative, sha256_file(str(ROOT / relative)))
    assert freeze["scientific_code_commit"] == (
        "dba612f4252b7552824737817db58e8fd0414cd9"
    )
    assert freeze["official_test_rows"] == 5621
    assert freeze["test_predictions_generated_at_freeze"] is False
    assert freeze["test_scores_observed_at_freeze"] is False
    assert freeze["execution"]["workers"] == 16
