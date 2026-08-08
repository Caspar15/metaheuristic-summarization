"""Run the preregistered S02b route-removal ablation on frozen dev only.

Only the dataset is selectable.  There is deliberately no split argument and
no test path.  The two candidates differ from the committed S02b config only
by removing semantic or graph from ``compute_budget.enabled_routes``.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from scripts.audit.run_greedy_sensitivity import (
    REPO_ROOT,
    _candidate_diagnostics,
    _load_study,
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
    "configs/preregistrations/d1_capacity_matched_route_ablation_v1.json"
)
DATASET_LABELS = {"multinews": "Multi-News", "govreport": "GovReport"}


def _load_preregistration() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("route-ablation preregistration must be an object")
    if value.get("partition") != "dev" or value.get("base_split") != "validation":
        raise ValueError("route ablation must remain bound to validation-dev")
    if value.get("dev_test_access") != "none":
        raise ValueError("route ablation must prohibit dev-test access")
    if not value.get("test_split_prohibited"):
        raise ValueError("route ablation must prohibit test access")
    return value


def _validate_base(config: dict[str, Any]) -> None:
    if config["compute_budget"]["enabled_routes"] != [
        "lexical",
        "semantic",
        "graph",
    ]:
        raise ValueError("ablation base must be the three-route S02b config")
    expected = {
        "route_top_k": 40,
        "min_per_route": 20,
        "total": 80,
    }
    actual = {key: int(config["candidate_budget"][key]) for key in expected}
    if actual != expected:
        raise ValueError(f"S02b candidate contract drift: {actual!r}")
    if int(config["coverage_guard"].get("max_items", -1)) != 20:
        raise ValueError("S02b coverage guard cap drift")
    if config["selector"].get("salience_source") != "rrf_fusion":
        raise ValueError("S02b selector salience drift")
    if config["optimizer"].get("method") != "greedy":
        raise ValueError("route ablation must use Greedy")


def run(dataset: str) -> dict[str, Any]:
    if dataset not in DATASET_LABELS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    label = DATASET_LABELS[dataset]
    preregistration = _load_preregistration()
    dataset_registration = preregistration["datasets"][label]
    base_path = REPO_ROOT / dataset_registration["base_config"]
    if sha256_file(str(base_path)) != dataset_registration["base_config_sha256"]:
        raise ValueError(f"{label} committed S02b config hash drift")
    base = load_yaml(str(base_path))
    _validate_base(base)

    spec = _load_study(dataset)
    partition = spec["manifest_object"]["partitions"]["dev"]
    ordered_ids = partition["selected_ids"]
    if len(ordered_ids) != int(dataset_registration["rows"]):
        raise ValueError(f"{label} frozen dev row-count drift")
    if selected_ids_sha256(ordered_ids) != partition["selected_ids_sha256"]:
        raise ValueError(f"{label} frozen dev selected-ID digest drift")

    output_root = REPO_ROOT / "runs_v2" / "d1_capacity_matched_route_ablation" / dataset / "dev"
    if output_root.exists():
        raise ValueError(f"refusing to overwrite route ablation: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    input_path = REPO_ROOT / spec["input"]
    gold = _load_gold(input_path, ordered_ids)
    preregistration_sha256 = sha256_file(str(REPO_ROOT / PREREGISTRATION))
    results: dict[str, dict[str, Any]] = {}

    expected_routes = {
        "A01_without_semantic": ["lexical", "graph"],
        "A02_without_graph": ["lexical", "semantic"],
    }
    for candidate in preregistration["candidates"]:
        candidate_id = candidate["id"]
        routes = candidate["delta"].get("compute_budget.enabled_routes")
        if routes != expected_routes.get(candidate_id):
            raise ValueError(f"route-ablation delta drift for {candidate_id}")
        config = copy.deepcopy(base)
        config["compute_budget"]["enabled_routes"] = list(routes)
        logical_hash = _canonical_sha256(
            {
                "study_id": preregistration["study_id"],
                "dataset": label,
                "base_config_sha256": dataset_registration["base_config_sha256"],
                "candidate": candidate,
                "partition_selected_ids_sha256": partition["selected_ids_sha256"],
            }
        )
        config["study"] = {
            "study_id": preregistration["study_id"],
            "family": "capacity_matched_route_ablation",
            "variant": candidate_id,
            "candidate_hash": logical_hash,
            "partition": "dev",
            "preregistration_path": PREREGISTRATION.as_posix(),
            "preregistration_sha256": preregistration_sha256,
        }
        candidate_root = output_root / candidate_id
        candidate_root.mkdir(parents=True, exist_ok=False)
        config_path = candidate_root / "resolved_config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )
        config_sha256 = sha256_file(str(config_path))
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
            "family": "capacity_matched_route_ablation",
            "candidate": candidate_id,
            "candidate_hash": logical_hash,
            "declared_delta": candidate["delta"],
            "base_candidate": "S02b_three_route_capacity_80",
        }
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
            diagnostics = _candidate_diagnostics(
                candidate_root / "greedy" / "run" / "predictions.jsonl"
            )
            if diagnostics["selector_candidate_size_max"] > 80:
                raise ValueError("capacity-matched ablation exceeded total cap 80")
            evidence_path = candidate_root / "greedy" / "run" / "evidence.json"
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
        _write_json(candidate_root / "study_summary.json", result)
        _append_search_log(
            {
                "logged_at_utc": _utc_now(),
                "study_id": preregistration["study_id"],
                "dataset": label,
                "partition": "dev",
                "family": "capacity_matched_route_ablation",
                "candidate": candidate_id,
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
                "reason": "capacity-matched route-removal ablation; pending paired analysis",
                "comparison_family_size": preregistration[
                    "comparison_family_size_after_ablation"
                ],
                "test_split_accessed": False,
            }
        )
        results[candidate_id] = result

    summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": (
            "completed"
            if all(result["status"] == "completed" for result in results.values())
            else "partial_failure"
        ),
        "study_id": preregistration["study_id"],
        "dataset": label,
        "partition": "dev",
        "partition_rows": len(ordered_ids),
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": preregistration_sha256,
        "candidates": results,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(output_root / "study_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LABELS))
    args = parser.parse_args()
    print(json.dumps(run(args.dataset), ensure_ascii=False))


if __name__ == "__main__":
    main()
