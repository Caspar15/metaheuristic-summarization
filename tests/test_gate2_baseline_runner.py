"""Structural tests for the dev-only Gate 2 baseline runner."""

import json
from pathlib import Path

import pytest

from scripts.audit import run_gate2_baseline_matrix as runner
from scripts.audit.run_length_contract_study import _embedding_cache_summary


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


def test_execution_cache_is_not_part_of_scientific_candidate_hash():
    protocol = _raw_protocol()
    variant = runner.expand_variants(protocol, "plm")[0]
    base = {"routes": {"semantic": {"model_name": "m", "revision": "r"}}}
    spec = {
        "manifest": "frozen-dev.json",
        "manifest_sha256": "a" * 64,
    }
    resolved = runner._resolved_config(
        base,
        spec=spec,
        family="plm",
        variant=variant,
        candidate_hash="candidate-hash",
    )
    assert "embedding_cache" not in resolved
    assert "embedding_cache_dir" not in resolved["routes"]["semantic"]


def test_embedding_cache_summary_counts_all_rows_and_hashes_order():
    rows = [
        {
            "id": "a",
            "baseline_diagnostics": {
                "representation": {
                    "embedding_cache": {
                        "status": "miss_written",
                        "cache_key": "1" * 64,
                        "contract_version": "v1",
                    }
                }
            },
        },
        {
            "id": "b",
            "baseline_diagnostics": {
                "representation": {
                    "embedding_cache": {
                        "status": "hit",
                        "cache_key": "2" * 64,
                        "contract_version": "v1",
                    }
                }
            },
        },
    ]
    summary = _embedding_cache_summary(rows)
    assert summary is not None
    assert summary["rows"] == 2
    assert summary["status_counts"] == {"miss_written": 1, "hit": 1}
    assert len(summary["ordered_row_cache_keys_sha256"]) == 64


def test_embedding_cache_summary_rejects_partial_provenance():
    rows = [
        {
            "id": "a",
            "baseline_diagnostics": {
                "representation": {
                    "embedding_cache": {
                        "status": "hit",
                        "cache_key": "1" * 64,
                        "contract_version": "v1",
                    }
                }
            },
        },
        {"id": "b", "baseline_diagnostics": {}},
    ]
    with pytest.raises(ValueError, match="only some rows"):
        _embedding_cache_summary(rows)


def test_embedding_cache_summary_accepts_pipeline_selector_provenance():
    cache = {
        "status": "hit",
        "cache_key": "a" * 64,
        "contract_version": "selector-cache-v1",
    }
    rows = [
        {
            "id": "row-1",
            "selector_inputs": {"representation": {"embedding_cache": cache}},
            "optimizer_diagnostics": {
                "selector_representation": {"embedding_cache": cache}
            },
        }
    ]
    summary = _embedding_cache_summary(rows)
    assert summary["rows"] == 1
    assert summary["status_counts"] == {"hit": 1}
    assert summary["contract_version"] == "selector-cache-v1"


def test_embedding_cache_summary_accepts_semantic_route_provenance():
    cache = {
        "status": "hit",
        "cache_key": "b" * 64,
        "contract_version": "route-cache-v1",
    }
    rows = [
        {
            "id": "row-1",
            "candidate_records": [
                {
                    "route_scores": {
                        "semantic": {"metadata": {"embedding_cache": cache}}
                    }
                },
                {
                    "route_scores": {
                        "semantic": {"metadata": {"embedding_cache": cache}}
                    }
                },
            ],
        }
    ]
    summary = _embedding_cache_summary(rows)
    assert summary["rows"] == 1
    assert summary["status_counts"] == {"hit": 1}
    assert summary["contract_version"] == "route-cache-v1"


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
