"""Run the preregistered capacity-correct three-route follow-up on dev.

Only the dataset is selectable.  The base split and partition are hard-bound
to frozen validation-dev manifests; there is no split CLI and no test path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.audit.run_greedy_sensitivity import (
    REPO_ROOT,
    _candidate_diagnostics,
    _load_protocol,
    _load_study,
    resolve_variant,
)
from scripts.audit.run_length_contract_study import (
    _append_search_log,
    _canonical_sha256,
    _load_gold,
    _relative,
    _run_method,
    _utc_now,
    _write_json,
)
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file
from src.utils.io import load_yaml


PREREGISTRATION = Path(
    "configs/preregistrations/d1_three_route_capacity_followup_v1.json"
)
DATASET_LABELS = {"multinews": "Multi-News", "govreport": "GovReport"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def run(dataset: str) -> dict[str, Any]:
    if dataset not in DATASET_LABELS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    label = DATASET_LABELS[dataset]
    preregistration = _load_json(PREREGISTRATION)
    if preregistration.get("partition") != "dev":
        raise ValueError("three-route follow-up must remain bound to dev")
    if preregistration.get("base_split") != "validation":
        raise ValueError("three-route follow-up must use validation")
    if label not in preregistration.get("datasets", []):
        raise ValueError(f"dataset {label!r} was not preregistered")
    candidate = preregistration["candidate"]
    expected_delta = {
        "candidate_budget.total": 80,
        "coverage_guard.max_items": 20,
    }
    if candidate.get("delta") != expected_delta:
        raise ValueError("three-route capacity delta drift")

    _, variants = _load_protocol()
    spec = _load_study(dataset)
    base = load_yaml(str(REPO_ROOT / spec["base_config"]))
    config = resolve_variant(
        base,
        variants,
        preregistration["parent_variant"],
        label,
    )
    routes = config["compute_budget"]["enabled_routes"]
    if routes != ["lexical", "semantic", "graph"]:
        raise ValueError("follow-up parent must enable lexical, semantic, graph")
    if int(config["candidate_budget"]["min_per_route"]) != 20:
        raise ValueError("follow-up derivation requires min_per_route 20")
    config["candidate_budget"]["total"] = 80
    config["coverage_guard"]["max_items"] = 20

    manifest = spec["manifest_object"]
    partition = manifest["partitions"]["dev"]
    ordered_ids = partition["selected_ids"]
    expected_rows = int(preregistration["success_contract"]["expected_rows"][label])
    if len(ordered_ids) != expected_rows:
        raise ValueError(f"{label} frozen dev row-count drift")
    if selected_ids_sha256(ordered_ids) != partition["selected_ids_sha256"]:
        raise ValueError(f"{label} selected-ID digest drift")

    preregistration_path = REPO_ROOT / PREREGISTRATION
    preregistration_sha256 = sha256_file(str(preregistration_path))
    logical_hash = _canonical_sha256(
        {
            "study_id": preregistration["study_id"],
            "dataset": label,
            "candidate": candidate,
            "parent_variant": preregistration["parent_variant"],
            "original_preregistration_sha256": sha256_file(
                str(
                    REPO_ROOT
                    / "configs/preregistrations/d1_greedy_sensitivity_v1.json"
                )
            ),
            "partition_selected_ids_sha256": partition["selected_ids_sha256"],
        }
    )
    config["experiment_partition"] = {
        "manifest_path": spec["manifest"],
        "manifest_sha256": spec["manifest_sha256"],
        "name": "dev",
    }
    config["study"] = {
        "study_id": preregistration["study_id"],
        "family": "semantic_route_followup",
        "variant": candidate["id"],
        "candidate_hash": logical_hash,
        "partition": "dev",
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": preregistration_sha256,
    }

    output_root = (
        REPO_ROOT
        / "runs_v2"
        / "d1_three_route_followup"
        / dataset
        / "dev"
        / candidate["id"]
    )
    if output_root.exists():
        raise ValueError(f"refusing to overwrite follow-up: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    config_path = output_root / "resolved_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    config_sha256 = sha256_file(str(config_path))

    input_path = REPO_ROOT / spec["input"]
    gold = _load_gold(input_path, ordered_ids)
    context = {
        "study_id": preregistration["study_id"],
        "dataset": label,
        "partition": "dev",
        "partition_rows": len(ordered_ids),
        "partition_manifest_path": spec["manifest"],
        "partition_manifest_sha256": spec["manifest_sha256"],
        "partition_selected_ids_sha256": partition["selected_ids_sha256"],
        "length_policy_path": spec["length_policy"],
        "length_policy_sha256": spec["length_policy_sha256"],
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": preregistration_sha256,
        "family": "semantic_route_followup",
        "candidate": candidate["id"],
        "candidate_hash": logical_hash,
        "declared_delta": candidate["delta"],
        "supersedes_original_failure": False,
        "trigger_candidate": "S02_lexical_semantic_graph",
    }
    try:
        metrics, _ = _run_method(
            "greedy",
            config_path=config_path,
            config_sha256=config_sha256,
            input_path=input_path,
            candidate_root=output_root,
            ordered_ids=ordered_ids,
            gold=gold,
            study_context=context,
        )
        diagnostics = _candidate_diagnostics(
            output_root / "greedy" / "run" / "predictions.jsonl"
        )
        if diagnostics["selector_candidate_size_max"] > 80:
            raise ValueError("three-route follow-up exceeded total cap 80")
        evidence_path = output_root / "greedy" / "run" / "evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["candidate_diagnostics"] = diagnostics
        evidence["dev_test_accessed"] = False
        _write_json(evidence_path, evidence)
        result = {
            "evidence_schema_version": "1.0",
            "measured_at_utc": _utc_now(),
            "status": "completed",
            "test_split_accessed": False,
            "dev_test_accessed": False,
            **context,
            "config_path": _relative(config_path),
            "config_sha256": config_sha256,
            "metrics": metrics,
            "candidate_diagnostics": diagnostics,
            "interpretation": "capacity follow-up only; original S02 remains failed",
        }
    except Exception as error:
        result = {
            "evidence_schema_version": "1.0",
            "measured_at_utc": _utc_now(),
            "status": "failed",
            "test_split_accessed": False,
            "dev_test_accessed": False,
            **context,
            "config_path": _relative(config_path),
            "config_sha256": config_sha256,
            "failure": f"{type(error).__name__}: {error}",
        }
    _write_json(output_root / "study_summary.json", result)
    _append_search_log(
        {
            "logged_at_utc": _utc_now(),
            "study_id": preregistration["study_id"],
            "dataset": label,
            "partition": "dev",
            "family": "semantic_route_followup",
            "candidate": candidate["id"],
            "candidate_hash": logical_hash,
            "config_path": _relative(config_path),
            "config_hash": config_sha256,
            "run_attempt": "final",
            "dev_score": (
                float(result["metrics"]["macro_rouge"])
                if result["status"] == "completed"
                else None
            ),
            "dev_test_score": None,
            "status": result["status"],
            "promoted": None,
            "reason": "capacity follow-up only; original S02 remains failed",
            "comparison_family_size": preregistration[
                "multiple_comparison_count_after_followups"
            ],
            "test_split_accessed": False,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LABELS))
    args = parser.parse_args()
    print(json.dumps(run(args.dataset), ensure_ascii=False))


if __name__ == "__main__":
    main()
