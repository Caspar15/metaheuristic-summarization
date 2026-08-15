import json
from copy import deepcopy

import pytest

from scripts.audit.verify_govreport_freeze_package import (
    FreezePackageError,
    REPO_ROOT,
    _assert_decision_contract,
    _assert_evidence_contract,
    _assert_lock_contract,
    _require_sha,
    validate_freeze_package,
)


def _locked_evidence():
    return {
        "partition": "GovReport frozen dev only",
        "further_dev_test_access": "forbidden",
        "dev_test_accessed_by_this_study": False,
        "test_split_prohibited": True,
        "test_split_accessed": False,
    }


def _locked_final():
    return {
        "protected_split": {
            "canonical_policy_path": "configs/data_policies/govreport_test_v1.json",
            "execution_locked": True,
            "test_membership_accessed": False,
            "test_payload_accessed": False,
            "test_references_accessed": False,
            "test_scores_accessed": False,
        },
        "execution": {"one_shot": True, "test_split_accessed": False},
    }


def test_committed_govreport_freeze_package_is_internally_consistent():
    report = validate_freeze_package()
    assert report["status"] == "pass"
    assert report["primary_dataset"] == "GovReport"
    assert report["boundary_dataset"] == "Multi-News"
    assert report["protected_splits_unlocked"] is False
    assert report["test_split_accessed"] is True
    assert report["evidence_completion_status"] == "E1_E2_E3_complete"
    assert report["policy_sequence_status"] == "resolved_by_authorized_two_stage_addendum"
    assert report["ready_for_policy_materialization_authorization"] is True
    assert report["ready_for_final_freeze_signature"] is False
    assert report["human_signature_status"] == "reported_approved_by_requesting_author"
    assert report["test_policy_materialized"] is True
    assert report["ready_for_test"] is False
    assert report["local_evidence_status"] in {
        "complete",
        "deferred_missing_untracked_artifacts",
    }


def test_lock_contract_rejects_test_score_access():
    final = _locked_final()
    final["protected_split"]["test_scores_accessed"] = True
    with pytest.raises(FreezePackageError, match="test_scores_accessed"):
        _assert_lock_contract(_locked_evidence(), final, REPO_ROOT)


def test_lock_contract_rejects_execution_unlock():
    final = _locked_final()
    final["protected_split"]["execution_locked"] = False
    with pytest.raises(FreezePackageError, match="not execution-locked"):
        _assert_lock_contract(_locked_evidence(), final, REPO_ROOT)


def test_decision_contract_rejects_primary_dataset_drift():
    decision = {
        "status": "requesting_author_approved_teacher_signature_pending",
        "dataset_roles": {
            "primary_quality_domain": {"dataset": "GovReport"},
            "boundary_condition": {"dataset": "Multi-News"},
        },
        "protected_split_policy": {
            "further_dev_test_access": "forbidden",
            "test_policy_materialized": False,
            "test_split_accessed": False,
        },
        "approval": {"protected_splits_unlocked": False},
    }
    drifted = deepcopy(decision)
    drifted["dataset_roles"]["primary_quality_domain"]["dataset"] = "Multi-News"
    with pytest.raises(FreezePackageError, match="sole primary"):
        _assert_decision_contract(drifted)


def test_optional_untracked_evidence_can_be_explicitly_deferred(tmp_path):
    present = _require_sha(
        tmp_path,
        "missing.jsonl",
        "0" * 64,
        "gitignored dev artifact",
        allow_missing=True,
    )
    assert present is False


def test_strict_local_evidence_mode_rejects_missing_artifact(tmp_path):
    with pytest.raises(FreezePackageError, match="pinned file is missing"):
        _require_sha(
            tmp_path,
            "missing.jsonl",
            "0" * 64,
            "gitignored dev artifact",
        )


def test_evidence_contract_rejects_posthoc_claim_downgrade_drift():
    index = json.loads(
        (REPO_ROOT / "configs/preregistrations/govreport_pretest_evidence_index_v1.json")
        .read_text(encoding="utf-8")
    )
    loaded = {
        label: json.loads((REPO_ROOT / index[label]["path"]).read_text(encoding="utf-8"))
        for label in ("E1", "E2", "E3")
    }
    drifted = deepcopy(loaded["E3"])
    drifted["required_claim_downgrades"] = ["graph_quality_contribution"]
    with pytest.raises(FreezePackageError, match="requires a claim downgrade"):
        _assert_evidence_contract(index, loaded["E1"], loaded["E2"], drifted)
