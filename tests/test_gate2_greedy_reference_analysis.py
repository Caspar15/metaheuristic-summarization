"""Golden tests for preregistered greedy-reference analysis formulas."""

import pytest

from scripts.audit.analyze_gate2_greedy_reference import (
    _headroom_summary,
    _load_protocol,
    _recall_summary,
    build_parser,
)


def test_analysis_protocol_is_dev_only_and_cli_has_no_split():
    protocol = _load_protocol()
    assert protocol["partition"] == "dev"
    assert protocol["dev_test_accessed"] is False
    assert protocol["test_split_prohibited"] is True
    assert "split" not in {action.dest for action in build_parser()._actions}


def test_headroom_formula_is_metric_specific():
    lead = {"rouge1": 0.4, "rouge2": 0.1, "rougeLsum": 0.3}
    greedy = {"rouge1": 0.6, "rouge2": 0.3, "rougeLsum": 0.5}
    system = {"rouge1": 0.5, "rouge2": 0.0, "rougeLsum": 0.5}
    result = _headroom_summary(lead, greedy, {"system": system})["systems"]["system"]
    assert result["per_metric"]["rouge1"]["captured_fraction"] == pytest.approx(0.5)
    assert result["per_metric"]["rouge2"]["captured_fraction"] == pytest.approx(-0.5)
    assert result["per_metric"]["rougeLsum"]["captured_fraction"] == pytest.approx(1.0)
    assert result["mean_captured_fraction"] == pytest.approx(1 / 3)


def test_candidate_recall_golden_with_empty_greedy_row():
    greedy = [
        {"id": "a", "selected_indices": [1, 2]},
        {"id": "b", "selected_indices": []},
        {"id": "c", "selected_indices": [3]},
    ]
    proposed = [
        {
            "id": "a",
            "selected_indices": [1],
            "candidate_pool": {
                "route_top_k": 40,
                "total_cap": 80,
                "actual_size": 2,
                "route_proposals": {
                    "lexical": [{"original_index": 1}],
                    "semantic": [{"original_index": 2}],
                    "graph": [{"original_index": 2}],
                }
            },
            "candidate_records": [
                {"original_index": 1, "selected_by_routes": ["lexical"]},
                {"original_index": 2, "selected_by_routes": ["semantic", "graph"]},
            ],
        },
        {
            "id": "b",
            "selected_indices": [],
            "candidate_pool": {
                "route_top_k": 40,
                "total_cap": 80,
                "actual_size": 0,
                "route_proposals": {"lexical": [], "semantic": [], "graph": []}
            },
            "candidate_records": [],
        },
        {
            "id": "c",
            "selected_indices": [4],
            "candidate_pool": {
                "route_top_k": 40,
                "total_cap": 80,
                "actual_size": 1,
                "route_proposals": {
                    "lexical": [],
                    "semantic": [],
                    "graph": [{"original_index": 4}],
                }
            },
            "candidate_records": [
                {"original_index": 4, "selected_by_routes": ["graph"]}
            ],
        },
    ]
    result = _recall_summary(greedy, proposed, route_top_k=40, total_cap=80)
    assert result["empty_greedy_rows"] == 1
    assert result["greedy_selected_denominator"] == 3
    assert result["sets"]["union_pool"]["micro_recall"] == pytest.approx(2 / 3)
    assert result["sets"]["union_pool"]["macro_recall"] == pytest.approx(0.5)
    assert result["sets"]["selected_set"]["micro_recall"] == pytest.approx(1 / 3)
    assert result["exclusive_route_hits"] == {"lexical": 1, "semantic": 0, "graph": 0}
