import copy
import json

import pytest

from scripts.audit.build_manuscript_supplemental_evidence import _length_statistics
from scripts.audit.run_postfreeze_no_reservation_ablation import (
    _apply_variant,
    _candidate_stats,
)


def test_no_reservation_changes_only_the_route_minimum() -> None:
    anchor = {
        "candidate_budget": {"min_per_route": 20, "total": 80},
        "compute_budget": {"enabled_routes": ["lexical", "semantic", "graph"]},
        "candidates": {
            "route_weights": {"lexical": 0.5, "semantic": 1.0, "graph": 1.0}
        },
    }
    variant = copy.deepcopy(anchor)

    _apply_variant(variant, "no_reservation", "govreport")

    assert variant["candidate_budget"] == {"min_per_route": 0, "total": 80}
    assert variant["compute_budget"] == anchor["compute_budget"]
    assert variant["candidates"] == anchor["candidates"]


@pytest.mark.parametrize(
    ("dataset", "expected_weights"),
    [
        ("govreport", {"semantic": 1.0, "graph": 1.0}),
        ("multinews", {"semantic": 1.0, "graph": 2.0}),
    ],
)
def test_no_lexical_route_preserves_the_frozen_nonlexical_weights(
    dataset: str, expected_weights: dict[str, float]
) -> None:
    config = {
        "candidate_budget": {"min_per_route": 20, "total": 80},
        "compute_budget": {"enabled_routes": ["lexical", "semantic", "graph"]},
        "candidates": {
            "route_weights": {"lexical": 0.5, "semantic": 1.0, "graph": 9.0}
        },
    }

    _apply_variant(config, "no_lexical_route", dataset)

    assert config["compute_budget"]["enabled_routes"] == ["semantic", "graph"]
    assert config["candidates"]["route_weights"] == expected_weights
    assert config["candidate_budget"] == {"min_per_route": 20, "total": 80}


@pytest.mark.parametrize(
    ("dataset", "expected_weights"),
    [
        ("govreport", {"lexical": 0.0, "semantic": 1.0, "graph": 1.0}),
        ("multinews", {"lexical": 0.0, "semantic": 1.0, "graph": 2.0}),
    ],
)
def test_zero_lexical_weight_changes_only_fusion_weights(
    dataset: str, expected_weights: dict[str, float]
) -> None:
    anchor = {
        "candidate_budget": {"route_top_k": 40, "min_per_route": 20, "total": 80},
        "compute_budget": {"enabled_routes": ["lexical", "semantic", "graph"]},
        "candidates": {
            "route_weights": {"lexical": 0.5, "semantic": 1.0, "graph": 1.0}
        },
    }
    variant = copy.deepcopy(anchor)

    _apply_variant(variant, "zero_lexical_weight_exact_pool", dataset)

    assert variant["candidates"]["route_weights"] == expected_weights
    assert variant["candidate_budget"] == anchor["candidate_budget"]
    assert variant["compute_budget"] == anchor["compute_budget"]


def test_candidate_stats_distinguishes_reserved_and_selected_exclusive(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    row = {
        "selected_indices": [1, 2],
        "candidate_records": [
            {
                "original_index": 1,
                "route_agreement": 1,
                "inclusion_reasons": ["reserve:lexical"],
            },
            {
                "original_index": 2,
                "route_agreement": 2,
                "inclusion_reasons": ["rrf_fill"],
            },
            {
                "original_index": 3,
                "route_agreement": 1,
                "inclusion_reasons": ["reserve:graph"],
            },
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = _candidate_stats(path)

    assert result["rows"] == 1
    assert result["mean_candidate_count"] == 3
    assert result["exclusive_candidate_fraction"] == pytest.approx(2 / 3)
    assert result["selected_exclusive_fraction"] == pytest.approx(1 / 2)
    assert result["reserved_candidate_records"] == 2


def test_length_statistics_uses_population_sd_and_feasibility(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"summary": "one two", "selected_indices": [0], "feasible": True},
        {
            "summary": "one two three four",
            "selected_indices": [0, 1, 2],
            "feasible": False,
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = _length_statistics([path])

    assert result["observations"] == 2
    assert result["mean_words"] == 3
    assert result["sd_words_population"] == 1
    assert result["mean_sentences"] == 2
    assert result["sd_sentences_population"] == 1
    assert result["feasible_observations"] == 1
    assert result["infeasible_observations"] == 1
