"""Governance and golden tests for the Gate 2 paired finalist diagnostic."""

import json

import pytest

from scripts.audit.analyze_gate2_paired_finalists import (
    METRICS, _load_per_example, _load_protocol, _paired_outcomes,
)


def test_protocol_is_dev_only_and_has_frozen_64_endpoint_family():
    protocol = _load_protocol()
    assert protocol["partition"] == "dev"
    assert protocol["dev_test_accessed"] is False
    assert protocol["test_split_prohibited"] is True
    assert protocol["multiplicity"]["holm_family_size"] == 64


def test_per_example_loader_requires_exact_order_and_builds_macro(tmp_path):
    path = tmp_path / "per_example.jsonl"
    rows = [
        {"id": "a", "rouge1": 0.3, "rouge2": 0.2, "rougeLsum": 0.1},
        {"id": "b", "rouge1": 0.6, "rouge2": 0.3, "rougeLsum": 0.0},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    values, evidence = _load_per_example(path, ["a", "b"])
    assert values["macro_rouge"] == pytest.approx([0.2, 0.3])
    assert evidence["rows"] == 2
    with pytest.raises(ValueError, match="exact frozen-dev order"):
        _load_per_example(path, ["b", "a"])


def test_per_example_loader_rejects_nonfinite_metric(tmp_path):
    path = tmp_path / "per_example.jsonl"
    path.write_text('{"id":"a","rouge1":NaN,"rouge2":0.2,"rougeLsum":0.1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite rouge1"):
        _load_per_example(path, ["a"])


def test_paired_outcomes_are_directional_and_selection_corrected():
    proposed = {metric: [0.9, 0.8, 0.7] for metric in METRICS}
    baselines = {"base": {metric: [0.1, 0.2, 0.3] for metric in METRICS}}
    comparisons, raw = _paired_outcomes(
        proposed, baselines, n_resamples=100, seed=7, selection_opportunities=10,
    )
    assert set(raw) == {f"base:{metric}" for metric in METRICS}
    for endpoint in comparisons["base"]["metrics"].values():
        assert endpoint["mean_difference"] > 0
        assert endpoint["p_value_selection_bonferroni"] == min(
            1.0, endpoint["p_value_two_sided"] * 10
        )
