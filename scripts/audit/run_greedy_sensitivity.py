"""Run the preregistered D1 Greedy sensitivity screen on frozen dev only.

There is intentionally no split argument.  This runner can read only the
``dev`` membership of the two frozen validation manifests; it has no code
path for dev-test or test.  Each family is an independent resumable stage and
every completed or failed configuration is appended to ``search_log.jsonl``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import yaml

from scripts.audit.run_length_contract_study import (
    REPO_ROOT,
    _append_search_log,
    _canonical_sha256,
    _git_commit,
    _load_gold,
    _load_search_log,
    _relative,
    _run_method,
    _utc_now,
    _write_json,
)
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file
from src.utils.io import load_yaml, read_jsonl


PREREGISTRATION = "configs/preregistrations/d1_greedy_sensitivity_v1.json"
PREREGISTRATION_SHA256 = "c608f847d77ff51224dffba529e3856e8b6832ae35ad8ddf03260cc1c4703c8f"
STUDIES: dict[str, dict[str, Any]] = {
    "multinews": {
        "dataset_label": "Multi-News",
        "input": "data/processed/multi_news_validation_canonical.jsonl",
        "base_config": "configs/studies/d1/multinews_base.yaml",
        "manifest": "configs/validation_partitions/multinews_validation_dev_v1.json",
        "manifest_sha256": "e61405482cda203c0bd50dda3e958986b11a51b48124129e68617f97ce9e42ee",
        "length_policy": "configs/length_policies/multinews_v1.json",
        "length_policy_sha256": "995ef68530343b7777b225dae1cf3739afa73e2f8cbae4d4dc2197d5bb7a3145",
    },
    "govreport": {
        "dataset_label": "GovReport",
        "input": "data/processed/govreport_validation_canonical.jsonl",
        "base_config": "configs/studies/d1/govreport_base.yaml",
        "manifest": "configs/validation_partitions/govreport_validation_dev_v1.json",
        "manifest_sha256": "7a15ffbb87abe690fe4e72a1e0daf27bf34b3a3293371983ae8e362d06e2717e",
        "length_policy": "configs/length_policies/govreport_v1.json",
        "length_policy_sha256": "10927b0ed465beb8d35d051bb606b2957370544d938bd45c850fb74140cb8515",
    },
}


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: dict[str, Any] = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"cannot apply D1 delta {dotted!r}: {part!r} is absent")
        node = child
    leaf = parts[-1]
    if leaf not in node:
        raise ValueError(f"cannot apply D1 delta {dotted!r}: leaf is absent")
    node[leaf] = copy.deepcopy(value)


def _variant_index(preregistration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for family, variants in preregistration["families"].items():
        for raw in variants:
            variant = dict(raw)
            variant_id = variant["id"]
            if variant_id in result:
                raise ValueError(f"duplicate D1 variant ID {variant_id!r}")
            variant["family"] = family
            result[variant_id] = variant
    if len(result) != 27:
        raise ValueError(f"D1 preregistration must contain 27 variants, got {len(result)}")
    return result


def resolve_variant(
    base_config: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    variant_id: str,
    dataset_label: str,
    _stack: tuple[str, ...] = (),
) -> dict[str, Any]:
    if variant_id in _stack:
        raise ValueError(f"D1 parent cycle: {_stack + (variant_id,)!r}")
    variant = variants[variant_id]
    parent = variant.get("parent")
    config = (
        resolve_variant(
            base_config, variants, str(parent), dataset_label, _stack + (variant_id,)
        )
        if parent
        else copy.deepcopy(dict(base_config))
    )
    for dotted, value in dict(variant.get("delta", {})).items():
        _set_dotted(config, dotted, value)
    dataset_delta = variant.get("dataset_delta", {})
    if dataset_delta:
        if dataset_label not in dataset_delta:
            raise ValueError(
                f"variant {variant_id!r} has no delta for {dataset_label!r}"
            )
        for dotted, value in dict(dataset_delta[dataset_label]).items():
            _set_dotted(config, dotted, value)
    return config


def _load_protocol() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = REPO_ROOT / PREREGISTRATION
    if sha256_file(str(path)) != PREREGISTRATION_SHA256:
        raise ValueError("frozen D1 preregistration SHA-256 drifted")
    preregistration = json.loads(path.read_text(encoding="utf-8"))
    if preregistration.get("status") != "frozen_before_d1_scores":
        raise ValueError("D1 preregistration is not frozen before scores")
    if preregistration.get("dev_test_access") != "none_in_screening":
        raise ValueError("D1 screening must prohibit dev-test access")
    if not preregistration.get("test_split_prohibited"):
        raise ValueError("D1 screening must prohibit test access")
    return preregistration, _variant_index(preregistration)


def _load_study(dataset: str) -> dict[str, Any]:
    spec = copy.deepcopy(STUDIES[dataset])
    for key in ("manifest", "length_policy"):
        expected = spec[f"{key}_sha256"]
        actual = sha256_file(str(REPO_ROOT / spec[key]))
        if actual != expected:
            raise ValueError(f"frozen D1 {key} SHA-256 drifted")
    manifest = json.loads((REPO_ROOT / spec["manifest"]).read_text(encoding="utf-8"))
    length_policy = json.loads(
        (REPO_ROOT / spec["length_policy"]).read_text(encoding="utf-8")
    )
    if length_policy.get("test_split_accessed") is not False:
        raise ValueError("length policy is not explicitly test-blind")
    spec["manifest_object"] = manifest
    spec["length_policy_object"] = length_policy
    return spec


def _logical_hash(
    *, dataset: str, family: str, variant_id: str, resolved_without_partition: Mapping[str, Any]
) -> str:
    return _canonical_sha256(
        {
            "study_id": "d1-greedy-sensitivity-v1",
            "dataset": dataset,
            "family": family,
            "variant": variant_id,
            "selector": "greedy",
            "config": resolved_without_partition,
            "evaluation": "all-row multisentence_lsum macro ROUGE",
        }
    )


def _candidate_diagnostics(predictions_path: Path) -> dict[str, Any]:
    rows = list(read_jsonl(str(predictions_path)))
    selector_sizes: list[int] = []
    provenance_sizes: list[int] = []
    agreements: list[float] = []
    unique_candidates: dict[str, list[int]] = {}
    unique_selected: dict[str, list[int]] = {}
    for row in rows:
        records = list(row.get("candidate_records", []))
        provenance_sizes.append(
            int(row.get("candidate_pool", {}).get("actual_size", len(records)))
        )
        selector_inputs = row.get("selector_inputs", {})
        if "candidate_count" not in selector_inputs:
            raise ValueError("prediction is missing selector_inputs.candidate_count")
        selector_sizes.append(int(selector_inputs["candidate_count"]))
        if records:
            agreements.append(float(np.mean([r.get("route_agreement", 0) for r in records])))
        selected = set(row.get("selected_indices", []))
        routes = sorted(
            {
                route
                for record in records
                for route in record.get("selected_by_routes", [])
            }
        )
        for route in routes:
            exclusive = [
                record
                for record in records
                if record.get("selected_by_routes") == [route]
            ]
            unique_candidates.setdefault(route, []).append(len(exclusive))
            unique_selected.setdefault(route, []).append(
                sum(record.get("original_index") in selected for record in exclusive)
            )
    return {
        "diagnostics_schema_version": "2.0",
        "rows": len(rows),
        # Backward-compatible name now means the actual selector search space,
        # including the full source when candidate prefiltering is disabled.
        "candidate_size_mean": (
            float(np.mean(selector_sizes)) if selector_sizes else 0.0
        ),
        "selector_candidate_size_mean": (
            float(np.mean(selector_sizes)) if selector_sizes else 0.0
        ),
        "selector_candidate_size_p95": (
            float(np.percentile(selector_sizes, 95)) if selector_sizes else 0.0
        ),
        "selector_candidate_size_max": max(selector_sizes, default=0),
        "provenance_candidate_size_mean": (
            float(np.mean(provenance_sizes)) if provenance_sizes else 0.0
        ),
        # No route records means not applicable, not zero agreement.
        "route_agreement_mean": (
            float(np.mean(agreements)) if agreements else None
        ),
        "unique_candidate_mean_by_route": {
            route: float(np.mean(values)) for route, values in unique_candidates.items()
        },
        "unique_selected_mean_by_route": {
            route: float(np.mean(values)) for route, values in unique_selected.items()
        },
    }


def _log_result_once(
    *,
    preregistration: Mapping[str, Any],
    spec: Mapping[str, Any],
    family: str,
    variant_id: str,
    result: Mapping[str, Any],
    run_attempt: str = "final",
) -> None:
    """Append one D1 attempt without duplicating an already-recorded attempt."""

    existing = _load_search_log()
    if any(
        row.get("study_id") == preregistration["study_id"]
        and row.get("dataset") == spec["dataset_label"]
        and row.get("family") == family
        and row.get("candidate") == variant_id
        and row.get("candidate_hash") == result["candidate_hash"]
        and row.get("run_attempt", "final") == run_attempt
        for row in existing
    ):
        return
    completed_run = result["status"] == "completed"
    _append_search_log(
        {
            "logged_at_utc": _utc_now(),
            "study_id": preregistration["study_id"],
            "dataset": spec["dataset_label"],
            "partition": "dev",
            "family": family,
            "candidate": variant_id,
            "candidate_hash": result["candidate_hash"],
            "config_path": result["config_path"],
            "config_hash": result["config_sha256"],
            "run_attempt": run_attempt,
            "dev_score": (
                float(result["metrics"]["macro_rouge"])
                if completed_run
                else None
            ),
            "dev_test_score": None,
            "status": result["status"],
            "promoted": None,
            "reason": (
                "pending all-family and cross-dataset screening decision"
                if completed_run
                else result.get("failure")
            ),
            "comparison_family_size": int(
                preregistration["measurement"]["comparison_count"]
            ),
            "test_split_accessed": False,
        }
    )


def _archive_interrupted_run(
    candidate_root: Path,
    *,
    context: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Preserve an externally interrupted run before retrying it.

    A missing ``candidate_summary.json`` means the family process terminated
    outside its normal exception handler (for example, a job-runner timeout).
    The incomplete artifact is evidence, not disposable scratch data.
    """

    run_path = candidate_root / "greedy" / "run"
    if not run_path.exists():
        raise ValueError(f"cannot recover missing run directory: {run_path}")
    attempts_root = candidate_root / "greedy" / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_number = 1
    while (attempts_root / f"attempt_{attempt_number:02d}_interrupted").exists():
        attempt_number += 1
    attempt_id = f"attempt_{attempt_number:02d}_interrupted"
    archive_path = attempts_root / attempt_id
    shutil.move(str(run_path), str(archive_path))
    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "failed",
        "failure_type": "external_runner_interruption",
        "failure": (
            "Incomplete atomic artifact recovered after external runner "
            "termination; no final predictions or method evidence existed."
        ),
        "test_split_accessed": False,
        "dev_test_accessed": False,
        **dict(context),
        "config_path": _relative(config_path),
        "config_sha256": config_sha256,
        "run_attempt": attempt_id,
        "archived_run_path": _relative(archive_path),
    }
    _write_json(archive_path / "interruption_evidence.json", evidence)
    return attempt_id, evidence


def run_family(dataset: str, family: str, *, resume: bool = False) -> dict[str, Any]:
    preregistration, variants = _load_protocol()
    if family not in preregistration["families"]:
        raise ValueError(f"unknown D1 family {family!r}")
    spec = _load_study(dataset)
    base = load_yaml(str(REPO_ROOT / spec["base_config"]))
    policy = spec["length_policy_object"]
    length = base.get("length_control", {})
    if (
        int(length.get("min_words", -1)) != int(policy["requested_min_words"])
        or int(length.get("max_words", -1)) != int(policy["max_words"])
    ):
        raise ValueError("D1 base config does not match the frozen length policy")

    partition_entry = spec["manifest_object"]["partitions"]["dev"]
    ordered_ids = partition_entry["selected_ids"]
    if selected_ids_sha256(ordered_ids) != partition_entry["selected_ids_sha256"]:
        raise ValueError("frozen D1 dev selected-ID digest is inconsistent")
    input_path = REPO_ROOT / spec["input"]
    gold = _load_gold(input_path, ordered_ids)
    output_root = REPO_ROOT / "runs_v2" / "d1_greedy_sensitivity" / dataset / "dev" / family
    if output_root.exists() and not resume:
        raise ValueError(f"refusing to overwrite existing D1 family: {output_root}")
    output_root.mkdir(parents=True, exist_ok=resume)

    results: dict[str, dict[str, Any]] = {}
    for raw_variant in preregistration["families"][family]:
        variant_id = raw_variant["id"]
        resolved_without_partition = resolve_variant(
            base, variants, variant_id, spec["dataset_label"]
        )
        logical_hash = _logical_hash(
            dataset=dataset,
            family=family,
            variant_id=variant_id,
            resolved_without_partition=resolved_without_partition,
        )
        config = copy.deepcopy(resolved_without_partition)
        config["experiment_partition"] = {
            "manifest_path": spec["manifest"],
            "manifest_sha256": spec["manifest_sha256"],
            "name": "dev",
        }
        config["study"] = {
            "study_id": preregistration["study_id"],
            "family": family,
            "variant": variant_id,
            "candidate_hash": logical_hash,
            "partition": "dev",
            "preregistration_path": PREREGISTRATION,
            "preregistration_sha256": PREREGISTRATION_SHA256,
        }
        candidate_root = output_root / variant_id
        config_path = candidate_root / "resolved_config.yaml"
        expected_config_text = yaml.safe_dump(
            config, sort_keys=False, allow_unicode=True
        )
        if candidate_root.exists():
            if not resume:
                raise ValueError(f"refusing to overwrite D1 candidate: {candidate_root}")
            if not config_path.exists():
                raise ValueError(f"resume candidate has no resolved config: {config_path}")
            existing_config = load_yaml(str(config_path))
            if existing_config != config:
                raise ValueError(f"resume config drift for D1 candidate {variant_id!r}")
        else:
            candidate_root.mkdir(parents=True, exist_ok=False)
            config_path.write_text(
                expected_config_text,
                encoding="utf-8",
                newline="\n",
            )
        config_sha256 = sha256_file(str(config_path))
        context = {
            "study_id": preregistration["study_id"],
            "dataset": spec["dataset_label"],
            "partition": "dev",
            "partition_rows": len(ordered_ids),
            "partition_manifest_path": spec["manifest"],
            "partition_manifest_sha256": spec["manifest_sha256"],
            "partition_selected_ids_sha256": partition_entry["selected_ids_sha256"],
            "length_policy_path": spec["length_policy"],
            "length_policy_sha256": spec["length_policy_sha256"],
            "preregistration_path": PREREGISTRATION,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "family": family,
            "candidate": variant_id,
            "candidate_hash": logical_hash,
            "declared_delta": raw_variant.get("delta", raw_variant.get("dataset_delta", {})),
        }
        candidate_summary_path = candidate_root / "candidate_summary.json"
        if candidate_summary_path.exists():
            result = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
            if (
                result.get("candidate_hash") != logical_hash
                or result.get("config_sha256") != config_sha256
                or result.get("candidate") != variant_id
            ):
                raise ValueError(f"resume summary drift for D1 candidate {variant_id!r}")
            if result.get("status") == "completed":
                evidence_path = candidate_root / "greedy" / "run" / "evidence.json"
                if not evidence_path.exists():
                    raise ValueError(
                        f"completed resume candidate has no evidence: {variant_id!r}"
                    )
                predictions_path = (
                    candidate_root / "greedy" / "run" / "predictions.jsonl"
                )
                diagnostics = _candidate_diagnostics(predictions_path)
                result["candidate_diagnostics"] = diagnostics
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                evidence["candidate_diagnostics"] = diagnostics
                evidence["candidate_diagnostics_implementation_commit"] = _git_commit()
                _write_json(evidence_path, evidence)
                _write_json(candidate_summary_path, result)
            results[variant_id] = result
            _log_result_once(
                preregistration=preregistration,
                spec=spec,
                family=family,
                variant_id=variant_id,
                result=result,
            )
            continue

        if candidate_root.exists() and (candidate_root / "greedy" / "run").exists():
            attempt_id, interrupted = _archive_interrupted_run(
                candidate_root,
                context=context,
                config_path=config_path,
                config_sha256=config_sha256,
            )
            _log_result_once(
                preregistration=preregistration,
                spec=spec,
                family=family,
                variant_id=variant_id,
                result=interrupted,
                run_attempt=attempt_id,
            )
        try:
            metrics, _ = _run_method(
                "greedy",
                config_path=config_path,
                config_sha256=config_sha256,
                input_path=input_path,
                candidate_root=candidate_root,
                ordered_ids=ordered_ids,
                gold=gold,
                study_context=context,
            )
            predictions_path = candidate_root / "greedy" / "run" / "predictions.jsonl"
            diagnostics = _candidate_diagnostics(predictions_path)
            evidence_path = candidate_root / "greedy" / "run" / "evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["candidate_diagnostics"] = diagnostics
            _write_json(evidence_path, evidence)
            result = {
                "status": "completed",
                **context,
                "config_path": _relative(config_path),
                "config_sha256": config_sha256,
                "metrics": metrics,
                "candidate_diagnostics": diagnostics,
            }
        except Exception as error:
            result = {
                "status": "failed",
                **context,
                "config_path": _relative(config_path),
                "config_sha256": config_sha256,
                "failure": f"{type(error).__name__}: {error}",
            }
        _write_json(candidate_summary_path, result)
        results[variant_id] = result
        _log_result_once(
            preregistration=preregistration,
            spec=spec,
            family=family,
            variant_id=variant_id,
            result=result,
        )

    completed = {
        name: row for name, row in results.items() if row["status"] == "completed"
    }
    ranking = sorted(
        completed,
        key=lambda name: (-float(completed[name]["metrics"]["macro_rouge"]), name),
    )
    summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed" if len(completed) == len(results) else "partial_failure",
        "test_split_accessed": False,
        "dev_test_accessed": False,
        "study_id": preregistration["study_id"],
        "dataset": spec["dataset_label"],
        "partition": "dev",
        "rows": len(ordered_ids),
        "family": family,
        "candidate_count": len(results),
        "ranking": ranking,
        "candidates": results,
        "promotion_status": "pending_all_families_and_both_datasets",
    }
    _write_json(output_root / "study_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(STUDIES))
    parser.add_argument(
        "--family",
        required=True,
        choices=("lexical_objective", "cheap_multiroute", "semantic_route"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "validate and reuse completed candidates, archive an externally "
            "interrupted attempt, and run only missing candidates"
        ),
    )
    args = parser.parse_args()
    summary = run_family(args.dataset, args.family, resume=args.resume)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "dataset": summary["dataset"],
                "partition": summary["partition"],
                "family": summary["family"],
                "ranking": summary["ranking"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
