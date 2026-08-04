"""Integrity tests for paired feasible-intersection evaluation."""

import json
import sys

import pytest

from scripts.audit.paired_run_intersection import (
    _load_run,
    _validate_run_against_gold,
    main,
)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_load_run_rejects_duplicate_prediction_ids(tmp_path):
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "a", "summary": "first", "feasible": True},
            {"id": "a", "summary": "second", "feasible": False},
        ],
    )

    with pytest.raises(ValueError, match="duplicate prediction id"):
        _load_run(str(path), assume_legacy_feasible=False)


def test_post_f17_run_must_cover_every_gold_id(tmp_path):
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(path, [{"id": "a", "summary": "a", "feasible": True}])
    run = _load_run(str(path), assume_legacy_feasible=False)

    with pytest.raises(ValueError, match="post-F-17 artifact is incomplete"):
        _validate_run_against_gold(run, {"a", "b"})


def test_legacy_run_may_expose_missing_ids_only_with_explicit_assumption(tmp_path):
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(path, [{"id": "a", "summary": "a"}])
    run = _load_run(str(path), assume_legacy_feasible=True)

    assert _validate_run_against_gold(run, {"a", "b"}) == ["b"]
    assert run["is_legacy"] is True


def test_load_run_preserves_infeasible_reason_for_report(tmp_path):
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "a",
                "summary": "short",
                "feasible": False,
                "infeasible_code": "selector_min_words_shortfall",
                "infeasible_reason": "five words short",
                "violations": {"min_words": 5.0},
            }
        ],
    )

    run = _load_run(str(path), assume_legacy_feasible=False)

    assert run["infeasible"] == [
        {
            "id": "a",
            "infeasible_code": "selector_min_words_shortfall",
            "infeasible_reason": "five words short",
            "violations": {"min_words": 5.0},
        }
    ]


def test_main_writes_paired_report_and_scores(tmp_path, monkeypatch):
    gold = tmp_path / "gold.jsonl"
    pred_a = tmp_path / "run-a" / "predictions.jsonl"
    pred_b = tmp_path / "run-b" / "predictions.jsonl"
    pred_a.parent.mkdir()
    pred_b.parent.mkdir()
    out_dir = tmp_path / "paired"
    _write_jsonl(
        gold,
        [
            {"id": "a", "highlights": "alpha reference"},
            {"id": "b", "highlights": "beta reference"},
        ],
    )
    _write_jsonl(
        pred_a,
        [
            {"id": "a", "summary": "alpha", "feasible": True},
            {
                "id": "b",
                "summary": "beta",
                "feasible": False,
                "infeasible_code": "selector_min_words_shortfall",
                "infeasible_reason": "short",
                "violations": {"min_words": 1.0},
            },
        ],
    )
    _write_jsonl(
        pred_b,
        [
            {"id": "a", "summary": "alpha", "feasible": True},
            {"id": "b", "summary": "beta", "feasible": True},
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "paired_run_intersection",
            "--pred",
            str(pred_a),
            str(pred_b),
            "--gold",
            str(gold),
            "--out_dir",
            str(out_dir),
            "--protocol",
            "multisentence_lsum",
        ],
    )

    main()

    report = json.loads(
        (out_dir / "intersection_report.json").read_text(encoding="utf-8")
    )
    assert report["intersection_size"] == 1
    assert report["runs"][0]["infeasible"][0]["id"] == "b"
    assert report["runs"][1]["excluded_feasible_from_intersection"] == ["b"]
    assert json.loads((out_dir / "include_ids.json").read_text(encoding="utf-8")) == ["a"]
    assert (out_dir / "run-a" / "per_example.jsonl").exists()
    assert (out_dir / "run-b" / "metrics.csv").exists()
