import copy
import json
from pathlib import Path

import pytest

from scripts.audit.run_greedy_sensitivity import (
    _archive_interrupted_run,
    _candidate_diagnostics,
    _load_protocol,
    _logical_hash,
    _set_dotted,
    resolve_variant,
)
from src.utils.io import load_yaml


def test_preregistered_variant_count_and_no_holdout_access():
    preregistration, variants = _load_protocol()
    assert len(variants) == 27
    assert preregistration["dev_test_access"] == "none_in_screening"
    assert preregistration["test_split_prohibited"] is True


def test_parent_delta_resolution_is_one_factor_from_graph_parent():
    preregistration, variants = _load_protocol()
    base = load_yaml("configs/studies/d1/multinews_base.yaml")
    graph = resolve_variant(base, variants, "G00_lexical_graph", "Multi-News")
    top_k = resolve_variant(base, variants, "G02_route_top_k_80", "Multi-News")
    expected = copy.deepcopy(graph)
    expected["candidate_budget"]["route_top_k"] = 80
    assert top_k == expected


def test_dataset_specific_structure_guard_resolves_differently():
    _, variants = _load_protocol()
    mn_base = load_yaml("configs/studies/d1/multinews_base.yaml")
    gov_base = load_yaml("configs/studies/d1/govreport_base.yaml")
    mn = resolve_variant(mn_base, variants, "G11_dataset_structure_guard", "Multi-News")
    gov = resolve_variant(gov_base, variants, "G11_dataset_structure_guard", "GovReport")
    assert mn["coverage_guard"]["document"] is False
    assert gov["coverage_guard"]["section"] is True


def test_dotted_delta_rejects_unknown_leaf():
    with pytest.raises(ValueError, match="leaf is absent"):
        _set_dotted({"a": {"b": 1}}, "a.typo", 2)


def test_logical_hash_is_partition_independent_and_delta_sensitive():
    _, variants = _load_protocol()
    base = load_yaml("configs/studies/d1/multinews_base.yaml")
    left = resolve_variant(base, variants, "L00_base", "Multi-News")
    right = resolve_variant(base, variants, "L01_importance_mean", "Multi-News")
    left_hash = _logical_hash(
        dataset="multinews", family="lexical_objective",
        variant_id="L00_base", resolved_without_partition=left,
    )
    right_hash = _logical_hash(
        dataset="multinews", family="lexical_objective",
        variant_id="L01_importance_mean", resolved_without_partition=right,
    )
    assert left_hash != right_hash


def test_archive_interrupted_run_preserves_partial_and_writes_evidence(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "scripts.audit.run_length_contract_study.REPO_ROOT", tmp_path
    )
    candidate = tmp_path / "L00_base"
    run_path = candidate / "greedy" / "run"
    run_path.mkdir(parents=True)
    (run_path / "predictions.jsonl.partial").write_text("partial\n", encoding="utf-8")
    config_path = candidate / "resolved_config.yaml"
    config_path.write_text("seed: 3407\n", encoding="utf-8")

    attempt_id, evidence = _archive_interrupted_run(
        candidate,
        context={
            "study_id": "d1-greedy-sensitivity-v1",
            "dataset": "Multi-News",
            "partition": "dev",
            "family": "lexical_objective",
            "candidate": "L00_base",
            "candidate_hash": "logical",
        },
        config_path=config_path,
        config_sha256="config",
    )

    archive = candidate / "greedy" / "attempts" / attempt_id
    assert not run_path.exists()
    assert (archive / "predictions.jsonl.partial").read_text(encoding="utf-8") == "partial\n"
    persisted = json.loads(
        (archive / "interruption_evidence.json").read_text(encoding="utf-8")
    )
    assert persisted == evidence
    assert evidence["status"] == "failed"
    assert evidence["test_split_accessed"] is False
    assert evidence["dev_test_accessed"] is False


def test_candidate_diagnostics_distinguish_selector_from_provenance_pool(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {
            "selector_inputs": {"candidate_count": 80},
            "candidate_pool": {"actual_size": 0},
            "candidate_records": [],
            "selected_indices": [1],
        },
        {
            "selector_inputs": {"candidate_count": 20},
            "candidate_pool": {"actual_size": 20},
            "candidate_records": [],
            "selected_indices": [2],
        },
    ]
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    diagnostics = _candidate_diagnostics(predictions)

    assert diagnostics["candidate_size_mean"] == 50.0
    assert diagnostics["selector_candidate_size_mean"] == 50.0
    assert diagnostics["selector_candidate_size_max"] == 80
    assert diagnostics["provenance_candidate_size_mean"] == 10.0
    assert diagnostics["route_agreement_mean"] is None
