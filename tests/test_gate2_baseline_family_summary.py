"""Unit tests for Gate 2 family interpretation helpers."""

import pytest

from scripts.audit.summarize_gate2_baseline_family import (
    _compare_selections,
    _prediction_diagnostics,
    _rank_results,
    build_parser,
)


def test_ranking_uses_macro_then_candidate_id():
    results = {
        "b": {"status": "completed", "method": "x", "metrics": {"macro_rouge": 0.5, "rouge": {}}},
        "a": {"status": "completed", "method": "x", "metrics": {"macro_rouge": 0.5, "rouge": {}}},
        "failed": {"status": "failed"},
    }
    assert [row["candidate"] for row in _rank_results(results)] == ["a", "b"]


def test_selection_comparison_is_id_aligned_and_order_sensitive():
    left = [
        {"id": "a", "selected_indices": [0, 1]},
        {"id": "b", "selected_indices": [2]},
    ]
    right = [
        {"id": "b", "selected_indices": [2]},
        {"id": "a", "selected_indices": [1, 0]},
    ]
    result = _compare_selections(left, right)
    assert result["exact_selected_indices_rows"] == 1
    assert result["mean_selected_indices_jaccard"] == pytest.approx(1.0)


def test_prediction_diagnostics_counts_score_degeneracy():
    result = _prediction_diagnostics(
        [
            {"baseline_diagnostics": {"score_degenerate": True}},
            {"baseline_diagnostics": {"score_degenerate": False}},
            {},
        ]
    )
    assert result["score_degenerate_rows"] == 1
    assert result["score_degenerate_rate"] == pytest.approx(1 / 3)


def test_cli_exposes_no_partition_argument():
    parser = build_parser()
    assert "partition" not in {action.dest for action in parser._actions}
