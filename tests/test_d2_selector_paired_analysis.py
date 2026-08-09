"""Governance tests for the frozen-dev D2 paired analyzer."""

import json

import pytest

from scripts.audit.analyze_d2_selector_full_dev import (
    METRICS,
    _load_per_example,
    _load_protocol,
    _paired_outcomes,
)


def test_protocol_freezes_dev_only_104_endpoint_family():
    protocol = _load_protocol()
    assert protocol["partition"] == "dev"
    assert protocol["dev_test_access"] == "none"
    assert protocol["test_split_prohibited"] is True
    assert len(protocol["candidates"]) == 14
    assert "104 Holm endpoints" in protocol["measurement"]["family"]


def test_per_example_loader_requires_exact_order_and_false_guards(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {"id": "a", "rouge1": 0.3, "rouge2": 0.2, "rougeLsum": 0.1},
        {"id": "b", "rouge1": 0.6, "rouge2": 0.3, "rougeLsum": 0.0},
    ]
    path = run_dir / "per_example.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (run_dir / "evidence.json").write_text(
        json.dumps({"dev_test_accessed": False, "test_split_accessed": False}),
        encoding="utf-8",
    )
    values, evidence = _load_per_example(path, ["a", "b"])
    assert values["macro_rouge"] == pytest.approx([0.2, 0.3])
    assert evidence["rows"] == 2
    with pytest.raises(ValueError, match="exact frozen-dev order"):
        _load_per_example(path, ["b", "a"])
    (run_dir / "evidence.json").write_text(
        json.dumps({"dev_test_accessed": True, "test_split_accessed": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="false dev-test guard"):
        _load_per_example(path, ["a", "b"])


def test_paired_outcomes_cover_non_anchor_family_and_correct_selection():
    scores = {
        "anchor": {metric: [0.1, 0.2, 0.3] for metric in METRICS},
        "better": {metric: [0.7, 0.8, 0.9] for metric in METRICS},
        "worse": {metric: [0.0, 0.1, 0.2] for metric in METRICS},
    }
    comparisons, raw = _paired_outcomes(
        scores, anchor_id="anchor", n_resamples=100, seed=7,
        selection_opportunities=228,
    )
    assert set(comparisons) == {"better", "worse"}
    assert len(raw) == 2 * len(METRICS)
    assert comparisons["better"]["metrics"]["macro_rouge"]["mean_difference"] > 0
    assert comparisons["worse"]["metrics"]["macro_rouge"]["mean_difference"] < 0
    for comparison in comparisons.values():
        for endpoint in comparison["metrics"].values():
            assert endpoint["p_value_selection_bonferroni"] == min(
                1.0, endpoint["p_value_two_sided"] * 228
            )
