"""Structural tests for the dev-only Gate 2 baseline runner."""

import json
from pathlib import Path

import pytest

from scripts.audit import run_gate2_baseline_matrix as runner


ROOT = Path(__file__).resolve().parents[1]


def _raw_protocol():
    return json.loads((ROOT / runner.PREREGISTRATION).read_text(encoding="utf-8"))


def test_preregistered_candidate_counts_and_unique_ids():
    protocol = _raw_protocol()
    non_plm = runner.expand_variants(protocol, "non_plm")
    plm = runner.expand_variants(protocol, "plm")
    assert len(non_plm) == 23
    assert len(plm) == 27
    assert len({item["id"] for item in non_plm + plm}) == 50


def test_pacsum_ofat_never_changes_two_axes_at_once():
    protocol = _raw_protocol()
    for family in ("non_plm", "plm"):
        variants = [v for v in runner.expand_variants(protocol, family) if v["method"].startswith("pacsum_")]
        for variant in variants:
            values = variant["pacsum"]
            changed_beta = values["beta"] != 0.0
            changed_direction = values["lambda_previous"] != 0.0
            assert not (changed_beta and changed_direction)
            assert values["lambda_following"] == pytest.approx(1.0 + values["lambda_previous"])


def test_cli_exposes_no_split_argument():
    parser = runner.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "split" not in destinations
    args = parser.parse_args(["--dataset", "multinews", "--family", "non_plm"])
    assert args.dataset == "multinews"
    assert args.family == "non_plm"


def test_protocol_explicitly_prohibits_devtest_and_test():
    protocol = _raw_protocol()
    assert protocol["frozen_partition"] == "dev"
    assert protocol["dev_test_access"] == "none_in_baseline_search"
    assert protocol["test_split_prohibited"] is True


def test_interrupted_resume_is_written_to_search_log(monkeypatch):
    captured = []
    monkeypatch.setattr(runner, "_load_search_log", lambda: [])
    monkeypatch.setattr(runner, "_append_search_log", captured.append)
    config_path = ROOT / "configs/preregistrations/gate2_baseline_matrix_v1.json"
    evidence = {
        "measured_at_utc": "2026-08-08T16:05:27+00:00",
        "failure_type": "external_runner_interruption",
        "failure": "Incomplete run preserved before an explicit --resume retry.",
        "archived_run_path": (
            "runs_v2/gate2_baseline_matrix_v1/multinews/dev/non_plm/"
            "candidate/method/attempts/attempt_01_interrupted"
        ),
    }

    runner._append_interruption_log(
        spec={"dataset_label": "Multi-News"},
        family="non_plm",
        variant={"id": "candidate", "method": "method"},
        candidate_hash="candidate-hash",
        config_path=config_path,
        config_sha256="config-hash",
        evidence=evidence,
    )

    assert len(captured) == 1
    row = captured[0]
    assert row["status"] == "failed"
    assert row["run_attempt"] == "attempt_01_interrupted"
    assert row["dev_score"] is None
    assert row["dev_test_accessed"] is False
    assert row["test_split_accessed"] is False
