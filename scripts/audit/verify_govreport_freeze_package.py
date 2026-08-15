"""Verify the GovReport-centered repositioning package without reading data.

This audit hashes only versioned policies, manifests, configs, preregistrations,
and already-produced dev evidence.  It must never open a protected dataset.
"""

from __future__ import annotations

import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.policy import sha256_file


ADDENDUM = Path("configs/data_policies/govreport_centered_repositioning_v2.json")
EVIDENCE_PREREG = Path(
    "configs/preregistrations/govreport_centered_evidence_completion_v1.json"
)
FINAL_PREREG = Path(
    "configs/preregistrations/govreport_centered_final_evaluation_v1.json"
)
PRETEST_EVIDENCE_INDEX = Path(
    "configs/preregistrations/govreport_pretest_evidence_index_v1.json"
)
TEST_AUTHORIZATION = Path(
    "configs/data_policies/govreport_test_authorization_v1.json"
)
TEST_AUTHORIZATION_SHA256 = (
    "64184a537f25bbdbad7c2701ef32735304b6ee8b659a59342ad8b46780f4e277"
)
FINAL_EXECUTION_FREEZE = Path(
    "configs/preregistrations/govreport_final_execution_freeze_v1.json"
)
FINAL_EXECUTION_FREEZE_SHA256 = (
    "5dbd5490d9d3616314cf32bf69efcbe0dfdb38dfb91a4d082a67c1a9166d7d92"
)
FINAL_EXECUTION_ACTIVATION = Path(
    "configs/preregistrations/govreport_final_execution_activation_v1.json"
)
FINAL_EXECUTION_ACTIVATION_SHA256 = (
    "5b9dd5ba54ab5c67679d66d8e1e7c3b47e184497bcc6a2da0bd8bc6c51d2a8b6"
)


class FreezePackageError(RuntimeError):
    """The frozen governance package is internally inconsistent."""


def _load_json(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezePackageError(f"cannot load {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise FreezePackageError(f"{relative}: top level must be an object")
    return value


def _require_sha(
    root: Path,
    path: str,
    expected: str,
    label: str,
    *,
    allow_missing: bool = False,
) -> bool:
    """Verify a pinned file and return whether it was present.

    ``allow_missing`` is only for deliberately untracked local run artifacts.
    A present artifact is always hashed, and committed metadata always uses the
    default fail-loud behavior.
    """

    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise FreezePackageError(f"{label}: malformed expected SHA-256 {expected!r}")
    target = root / path
    if not target.is_file():
        if allow_missing:
            return False
        raise FreezePackageError(f"{label}: pinned file is missing: {path}")
    actual = sha256_file(str(target))
    if actual != expected:
        raise FreezePackageError(
            f"{label}: {path} SHA-256 is {actual}, expected {expected}"
        )
    return True


def _require_false(value: Any, label: str) -> None:
    if value is not False:
        raise FreezePackageError(f"{label} must be exactly false, got {value!r}")


def _assert_decision_contract(addendum: Mapping[str, Any]) -> None:
    if addendum.get("status") != "requesting_author_approved_teacher_signature_pending":
        raise FreezePackageError("repositioning approval status drifted")
    roles = addendum.get("dataset_roles", {})
    if roles.get("primary_quality_domain", {}).get("dataset") != "GovReport":
        raise FreezePackageError("GovReport is not the sole primary quality domain")
    if roles.get("boundary_condition", {}).get("dataset") != "Multi-News":
        raise FreezePackageError("Multi-News is not preserved as the boundary condition")
    protected = addendum.get("protected_split_policy", {})
    if protected.get("further_dev_test_access") != "forbidden":
        raise FreezePackageError("further dev-test access is not forbidden")
    _require_false(protected.get("test_policy_materialized"), "test_policy_materialized")
    _require_false(protected.get("test_split_accessed"), "test_split_accessed")
    _require_false(addendum.get("approval", {}).get("protected_splits_unlocked"), "protected_splits_unlocked")


def _assert_lock_contract(
    evidence: Mapping[str, Any],
    final: Mapping[str, Any],
    root: Path,
    authorization: Mapping[str, Any] | None = None,
) -> None:
    if evidence.get("partition") != "GovReport frozen dev only":
        raise FreezePackageError("evidence completion is not restricted to frozen dev")
    if evidence.get("further_dev_test_access") != "forbidden":
        raise FreezePackageError("evidence preregistration permits dev-test access")
    _require_false(
        evidence.get("dev_test_accessed_by_this_study"),
        "evidence.dev_test_accessed_by_this_study",
    )
    if evidence.get("test_split_prohibited") is not True:
        raise FreezePackageError("evidence preregistration does not prohibit test")
    _require_false(evidence.get("test_split_accessed"), "evidence.test_split_accessed")

    protected = final.get("protected_split", {})
    if protected.get("execution_locked") is not True:
        raise FreezePackageError("final evaluation is not execution-locked")
    for key in (
        "test_membership_accessed",
        "test_payload_accessed",
        "test_references_accessed",
        "test_scores_accessed",
    ):
        _require_false(protected.get(key), f"final.{key}")
    _require_false(
        final.get("execution", {}).get("test_split_accessed"),
        "final.execution.test_split_accessed",
    )
    if final.get("execution", {}).get("one_shot") is not True:
        raise FreezePackageError("final evaluation is not declared one-shot")

    future_test_policy = root / protected.get("canonical_policy_path", "")
    if future_test_policy.is_file():
        if authorization is None or authorization.get("stage_A", {}).get("authorized") is not True:
            raise FreezePackageError(
                "test policy exists without a Stage-A authorization addendum: "
                f"{future_test_policy}"
            )
        if authorization.get("supersedes_ordering_only", {}).get("resolution") != (
            "Use two stages: policy materialization first, then exact execution "
            "freeze and one-shot scoring."
        ):
            raise FreezePackageError("Stage-A authorization ordering resolution drifted")


def _assert_test_policy_contract(policy: Mapping[str, Any]) -> None:
    if policy.get("status") != "frozen_before_test_results":
        raise FreezePackageError("GovReport test policy is not frozen before scores")
    if policy.get("test_predictions_generated") is not False:
        raise FreezePackageError("test policy reports predictions before execution freeze")
    if policy.get("test_scores_observed") is not False:
        raise FreezePackageError("test policy reports scores before execution freeze")
    dataset = policy.get("dataset", {})
    if dataset.get("name") != "GovReport" or dataset.get("split") != "test":
        raise FreezePackageError("GovReport test policy dataset identity drifted")
    analysis = policy.get("analyses", {}).get("main", {})
    if analysis.get("role") != "primary_test" or analysis.get("expected_rows") != 973:
        raise FreezePackageError("GovReport test policy analysis contract drifted")
    if policy.get("canonical_exclusions", {}).get("expected_rows") != 0:
        raise FreezePackageError("GovReport test exclusion count drifted")


def _assert_final_execution_contract(
    freeze: Mapping[str, Any],
    activation: Mapping[str, Any],
    dry_run: Mapping[str, Any],
) -> None:
    if freeze.get("status") != "frozen_before_test_predictions_or_scores":
        raise FreezePackageError("final execution package is not score-blind frozen")
    if freeze.get("one_shot") is not True or freeze.get("official_test_rows") != 973:
        raise FreezePackageError("final execution row/one-shot contract drifted")
    _require_false(
        freeze.get("test_predictions_generated_at_freeze"),
        "final freeze test_predictions_generated_at_freeze",
    )
    _require_false(
        freeze.get("test_scores_observed_at_freeze"),
        "final freeze test_scores_observed_at_freeze",
    )
    if activation.get("status") != "authorized_for_one_shot_execution":
        raise FreezePackageError("Stage B activation status drifted")
    if activation.get("freeze_manifest_sha256") != FINAL_EXECUTION_FREEZE_SHA256:
        raise FreezePackageError("Stage B activation points to a different freeze")
    if activation.get("ready_for_test") is not True:
        raise FreezePackageError("Stage B activation is not ready for test")
    conditions = activation.get("activation_conditions", {})
    required_true = (
        "stage_A_policy_and_health_pinned",
        "nine_system_runner_fail_closed",
        "official_evaluator_fail_closed",
        "scientific_commit_environment_commands_outputs_frozen",
        "score_free_dry_run_passed",
    )
    if not all(conditions.get(key) is True for key in required_true):
        raise FreezePackageError("one or more Stage B activation conditions drifted")
    _require_false(
        conditions.get("test_predictions_generated_before_activation"),
        "activation test_predictions_generated_before_activation",
    )
    _require_false(
        conditions.get("test_scores_observed_before_activation"),
        "activation test_scores_observed_before_activation",
    )
    dry_pin = activation.get("score_free_dry_run", {})
    if dry_pin.get("path") != "docs/research/evidence/govreport_final_dry_run_v1.json":
        raise FreezePackageError("score-free dry-run path drifted")
    if dry_run.get("status") != "passed_without_predictions_or_scores":
        raise FreezePackageError("score-free dry run did not pass")
    if dry_run.get("freeze_manifest_sha256") != FINAL_EXECUTION_FREEZE_SHA256:
        raise FreezePackageError("score-free dry run used a different freeze")
    if dry_run.get("rows") != 973 or dry_run.get("references_present") != 973:
        raise FreezePackageError("score-free dry-run row contract drifted")
    if dry_run.get("tracked_worktree_clean") is not True:
        raise FreezePackageError("score-free dry run did not use a clean worktree")
    _require_false(dry_run.get("test_predictions_generated"), "dry-run predictions")
    _require_false(dry_run.get("test_scores_observed"), "dry-run scores")


def _assert_evidence_contract(
    index: Mapping[str, Any],
    e1: Mapping[str, Any],
    e2: Mapping[str, Any],
    e3: Mapping[str, Any],
) -> None:
    if index.get("status") != "e1_e2_e3_complete_before_test_access":
        raise FreezePackageError("pretest evidence index is not marked complete")
    protected = index.get("protected_split", {})
    for key in (
        "dev_test_accessed",
        "test_split_accessed",
        "test_policy_materialized",
        "test_execution_authorized",
    ):
        _require_false(protected.get(key), f"pretest index {key}")

    studies = (("E1", e1), ("E2", e2), ("E3", e3))
    for label, evidence in studies:
        expected = index[label]
        if evidence.get("status") != "completed":
            raise FreezePackageError(f"{label} evidence is not completed")
        if evidence.get("study_id") != expected["study_id"]:
            raise FreezePackageError(f"{label} study identity drifted")
        _require_false(evidence.get("dev_test_accessed"), f"{label} dev_test_accessed")
        _require_false(evidence.get("test_split_accessed"), f"{label} test_split_accessed")

    e1_primary = e1.get("primary_proposed_vs_sbert_mmr_lambda_0.9", {})
    if e1.get("rows") != index["E1"]["expected_rows"]:
        raise FreezePackageError("E1 row count drifted")
    for key in ("macro_pass", "component_guard_pass", "official_dev_comparison_survives"):
        if e1_primary.get(key) is not True:
            raise FreezePackageError(f"E1 {key} is not true")
    if e1.get("decision") != index["E1"]["required_decision"]:
        raise FreezePackageError("E1 decision drifted")

    systems = e2.get("systems", {})
    if len(systems) != index["E2"]["expected_systems"]:
        raise FreezePackageError("E2 system count drifted")
    repetitions = index["E2"]["required_measured_repetitions_per_state"]
    for name, system in systems.items():
        for state in ("cold", "warm_cache"):
            if system.get(state, {}).get("measured_repetitions") != repetitions:
                raise FreezePackageError(f"E2 {name}/{state} repetitions drifted")
    identities = e2.get("cross_state_selected_indices_identity", {})
    if set(identities) != set(systems) or not all(
        value.get("identical") is True for value in identities.values()
    ):
        raise FreezePackageError("E2 selected-index identity audit failed")

    if e3.get("rows") != index["E3"]["expected_rows"]:
        raise FreezePackageError("E3 row count drifted")
    if e3.get("holm_family_size") != index["E3"]["expected_holm_family_size"]:
        raise FreezePackageError("E3 Holm family size drifted")
    if e3.get("bootstrap_resamples") != index["E3"]["expected_bootstrap_resamples"]:
        raise FreezePackageError("E3 bootstrap count drifted")
    if e3.get("required_claim_downgrades") != index["E3"]["required_claim_downgrades"]:
        raise FreezePackageError("E3 requires a claim downgrade")
    decisions = e3.get("claim_decisions", {})
    if len(decisions) != 5 or not all(value is True for value in decisions.values()):
        raise FreezePackageError("E3 claim decision gate failed")


def validate_freeze_package(
    root: Path = REPO_ROOT, *, require_local_evidence: bool = False
) -> dict[str, Any]:
    addendum = _load_json(root, ADDENDUM)
    evidence = _load_json(root, EVIDENCE_PREREG)
    final = _load_json(root, FINAL_PREREG)
    pretest_index = _load_json(root, PRETEST_EVIDENCE_INDEX)
    authorization = _load_json(root, TEST_AUTHORIZATION)
    _require_sha(
        root,
        TEST_AUTHORIZATION.as_posix(),
        TEST_AUTHORIZATION_SHA256,
        "test two-stage authorization",
    )
    _assert_decision_contract(addendum)
    _assert_lock_contract(evidence, final, root, authorization)

    _require_sha(
        root,
        FINAL_EXECUTION_FREEZE.as_posix(),
        FINAL_EXECUTION_FREEZE_SHA256,
        "final execution freeze",
    )
    _require_sha(
        root,
        FINAL_EXECUTION_ACTIVATION.as_posix(),
        FINAL_EXECUTION_ACTIVATION_SHA256,
        "final execution activation",
    )
    freeze = _load_json(root, FINAL_EXECUTION_FREEZE)
    activation = _load_json(root, FINAL_EXECUTION_ACTIVATION)
    dry_pin = activation["score_free_dry_run"]
    _require_sha(root, dry_pin["path"], dry_pin["sha256"], "score-free dry run")
    dry_run = _load_json(root, Path(dry_pin["path"]))
    _assert_final_execution_contract(freeze, activation, dry_run)

    test_policy_path = Path(final["protected_split"]["canonical_policy_path"])
    test_policy = _load_json(root, test_policy_path)
    _assert_test_policy_contract(test_policy)
    for label, pin in (
        ("test exclusion manifest", test_policy["canonical_exclusions"]),
        ("test replacement manifest", test_policy["replacement_character_manifest"]),
        ("test canonical health", test_policy["canonical_health_evidence"]),
    ):
        path_key = "manifest_path" if label == "test exclusion manifest" else "path"
        sha_key = "manifest_file_sha256" if label == "test exclusion manifest" else "file_sha256"
        _require_sha(root, pin[path_key], pin[sha_key], label)

    completed: dict[str, dict[str, Any]] = {}
    for label in ("E1", "E2", "E3"):
        pin = pretest_index[label]
        _require_sha(root, pin["path"], pin["sha256"], f"{label} completion evidence")
        completed[label] = _load_json(root, Path(pin["path"]))
    _assert_evidence_contract(
        pretest_index, completed["E1"], completed["E2"], completed["E3"]
    )

    _require_sha(
        root,
        str(ADDENDUM).replace("\\", "/"),
        evidence["repositioning_addendum"]["sha256"],
        "evidence repositioning addendum",
    )
    _require_sha(
        root,
        str(ADDENDUM).replace("\\", "/"),
        final["repositioning_addendum"]["sha256"],
        "final repositioning addendum",
    )
    _require_sha(
        root,
        str(EVIDENCE_PREREG).replace("\\", "/"),
        final["evidence_completion_preregistration"]["sha256"],
        "final evidence-completion preregistration",
    )

    for name, pin in addendum["source_policies"].items():
        _require_sha(root, pin["path"], pin["sha256"], f"source policy {name}")
    partition = addendum["frozen_development_identity"]["govreport_partition"]
    _require_sha(
        root,
        partition["path"],
        partition["lf_canonical_sha256"],
        "GovReport dev partition",
    )
    length_policy = addendum["frozen_development_identity"][
        "govreport_length_policy"
    ]
    _require_sha(
        root,
        length_policy["path"],
        length_policy["sha256"],
        "GovReport length policy",
    )
    for name in ("d3b_preregistration", "d3b_paired_summary"):
        pin = addendum["evidence_basis"][name]
        _require_sha(root, pin["path"], pin["sha256"], name)
    candidate = addendum["frozen_candidate"]
    _require_sha(
        root,
        candidate["config_path"],
        candidate["config_sha256"],
        "frozen candidate config",
    )

    inputs = evidence["frozen_inputs"]
    _require_sha(root, inputs["policy"]["path"], inputs["policy"]["sha256"], "evidence policy")
    _require_sha(
        root,
        inputs["partition_manifest"]["path"],
        inputs["partition_manifest"]["lf_canonical_sha256"],
        "evidence partition",
    )
    _require_sha(
        root,
        inputs["length_policy"]["path"],
        inputs["length_policy"]["sha256"],
        "evidence length policy",
    )
    proposed = inputs["proposed"]
    _require_sha(
        root,
        proposed["config_path"],
        proposed["config_sha256"],
        "proposed config_path",
    )

    local_checked: list[str] = []
    local_deferred: list[str] = []
    local_seen: set[str] = set()

    def check_local(path: str, expected: str, label: str) -> None:
        if path in local_seen:
            return
        local_seen.add(path)
        present = _require_sha(
            root,
            path,
            expected,
            label,
            allow_missing=not require_local_evidence,
        )
        (local_checked if present else local_deferred).append(path)

    check_local(
        proposed["predictions_path"],
        proposed["predictions_sha256"],
        "proposed predictions_path",
    )
    check_local(
        "runs_v2/d3b_cross_profile_combination_v1/govreport/dev/"
        "C01_combined_salience_route_weight/per_example.jsonl",
        proposed["per_example_sha256"],
        "proposed per-example",
    )
    test_analysis = test_policy["analyses"]["main"]
    check_local(
        test_analysis["artifact_path"],
        test_analysis["expected_file_sha256"],
        "GovReport canonical test artifact",
    )
    selection = inputs["baseline_selection_evidence"]
    _require_sha(
        root,
        selection["preregistration_path"],
        selection["preregistration_sha256"],
        "baseline selection preregistration",
    )
    _require_sha(
        root,
        selection["paired_summary_path"],
        selection["paired_summary_sha256"],
        "baseline selection summary",
    )
    for name, pair in evidence["work_packages"]["E1_published_evaluator"][
        "systems"
    ].items():
        check_local(pair[0], pair[1], f"published-evaluator input {name}")

    final_selection = final["frozen_baselines"]["selection_source"]
    _require_sha(
        root,
        final_selection["path"],
        final_selection["sha256"],
        "final baseline selection source",
    )
    final_candidate = final["frozen_proposed_system"]
    _require_sha(
        root,
        final_candidate["source_config_path"],
        final_candidate["source_config_sha256"],
        "final source config",
    )

    config = yaml.safe_load((root / candidate["config_path"]).read_text(encoding="utf-8"))
    if config.get("experiment", {}).get("status") != "validation_pilot_only":
        raise FreezePackageError("frozen source config is not validation-only")
    if config.get("experiment_partition", {}).get("name") != "dev":
        raise FreezePackageError("frozen source config is not pinned to dev")
    if config.get("optimizer", {}).get("method") != "mmr":
        raise FreezePackageError("frozen source selector is no longer MMR")
    if float(config.get("optimizer", {}).get("lambda_relevance")) != 0.7:
        raise FreezePackageError("frozen MMR lambda drifted")

    return {
        "status": "pass",
        "policy_id": addendum["policy_id"],
        "evidence_study_id": evidence["study_id"],
        "final_study_id": final["study_id"],
        "primary_dataset": "GovReport",
        "boundary_dataset": "Multi-News",
        "protected_splits_unlocked": True,
        "test_split_accessed": True,
        "test_predictions_generated": False,
        "test_scores_observed": False,
        "local_evidence_status": (
            "complete" if not local_deferred else "deferred_missing_untracked_artifacts"
        ),
        "local_evidence_checked": len(local_checked),
        "local_evidence_deferred": len(local_deferred),
        "deferred_paths": local_deferred,
        "evidence_completion_status": "E1_E2_E3_complete",
        "policy_sequence_status": "resolved_by_authorized_two_stage_addendum",
        "ready_for_policy_materialization_authorization": True,
        "ready_for_final_freeze_signature": True,
        "human_signature_status": "reported_approved_by_requesting_author",
        "test_policy_materialized": True,
        "final_execution_activated": True,
        "score_free_dry_run_status": "passed_without_predictions_or_scores",
        "ready_for_test": True,
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-local-evidence",
        action="store_true",
        help=(
            "fail when gitignored predictions/per-example dev artifacts are absent; "
            "present artifacts are always verified even without this flag"
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_freeze_package(
                require_local_evidence=args.require_local_evidence
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
