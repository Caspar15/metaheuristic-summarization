import copy

import pytest

from scripts.audit.run_d2_selector_full_dev import (
    _load_preregistration,
    _validate_base,
    resolve_candidate,
)
from scripts.audit.run_length_contract_study import _command_for_method
from src.utils.io import load_yaml


BASE = "runs_v2/d1_three_route_followup/multinews/dev/S02b_three_route_capacity_80/resolved_config.yaml"


def test_d2_preregistration_is_frozen_dev_only_and_complete():
    protocol = _load_preregistration()
    assert protocol["partition"] == "dev"
    assert protocol["dev_test_access"] == "none"
    assert protocol["test_split_prohibited"] is True
    assert len(protocol["candidates"]) == 14
    assert sum(bool(row.get("reuse_base_run")) for row in protocol["candidates"]) == 1


def test_d2_candidate_resolution_changes_only_declared_selector_fields():
    protocol = _load_preregistration()
    base = load_yaml(BASE)
    candidate = next(row for row in protocol["candidates"] if row["id"] == "S10_mmr_sbert_l07")
    resolved = resolve_candidate(base, candidate)
    assert resolved["optimizer"]["method"] == "mmr"
    assert resolved["optimizer"]["lambda_relevance"] == 0.7
    assert resolved["selector"]["similarity_source"] == "sbert"
    for key in ("compute_budget", "candidate_budget", "candidates", "coverage_guard", "length_control"):
        assert resolved[key] == base[key]


def test_d2_base_validation_rejects_candidate_budget_drift():
    base = load_yaml(BASE)
    _validate_base(base)
    bad = copy.deepcopy(base)
    bad["candidate_budget"]["total"] = 81
    with pytest.raises(ValueError, match="budget drift"):
        _validate_base(bad)


def test_pipeline_selector_dispatches_mmr_through_canonical_pipeline(tmp_path):
    command = _command_for_method(
        "mmr",
        config_path=tmp_path / "config.yaml",
        input_path=tmp_path / "input.jsonl",
        method_root=tmp_path / "run",
        pipeline_selector=True,
    )
    assert command[1:3] == ["-m", "src.pipeline.select_sentences"]
    assert "--baseline" not in command
