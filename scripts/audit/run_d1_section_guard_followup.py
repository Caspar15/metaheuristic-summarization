"""Run the preregistered GovReport section-guard overflow follow-up on dev.

The script has no split argument and is deliberately bound to the frozen
GovReport validation-dev manifest.  It preserves the original failed G11 run;
this is a separately registered feasibility follow-up, not a replacement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from scripts.audit.run_greedy_sensitivity import (
    REPO_ROOT,
    STUDIES,
    _candidate_diagnostics,
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
    "configs/preregistrations/d1_govreport_section_guard_followup_v1.json"
)
OUTPUT_ROOT = Path(
    "runs_v2/d1_section_guard_followup/govreport/dev/G11b_section_guard_cap20"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def run() -> dict[str, Any]:
    preregistration = _load_json(PREREGISTRATION)
    if preregistration.get("partition") != "dev":
        raise ValueError("follow-up preregistration must remain bound to dev")
    if preregistration.get("base_split") != "validation":
        raise ValueError("follow-up preregistration must use validation")
    candidate = preregistration["candidate"]
    if candidate.get("delta") != {"coverage_guard.max_items": 20}:
        raise ValueError("section-guard follow-up delta drift")

    spec = STUDIES["govreport"]
    manifest_path = REPO_ROOT / spec["manifest"]
    if sha256_file(str(manifest_path)) != spec["manifest_sha256"]:
        raise ValueError("GovReport frozen manifest hash drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partition = manifest["partitions"]["dev"]
    ordered_ids = partition["selected_ids"]
    if len(ordered_ids) != 681:
        raise ValueError("GovReport follow-up must contain 681 frozen dev rows")
    if selected_ids_sha256(ordered_ids) != partition["selected_ids_sha256"]:
        raise ValueError("GovReport follow-up selected-ID digest drift")

    parent_path = REPO_ROOT / preregistration["parent_config"]
    config = load_yaml(str(parent_path))
    if not bool(config["coverage_guard"].get("section")):
        raise ValueError("parent G11 config no longer enables section guard")
    if int(config["candidate_budget"]["total"]) != 60:
        raise ValueError("follow-up derivation requires candidate total 60")
    if int(config["candidate_budget"]["min_per_route"]) != 20:
        raise ValueError("follow-up derivation requires min_per_route 20")
    if len(config["compute_budget"]["enabled_routes"]) != 2:
        raise ValueError("follow-up derivation requires two active routes")
    config["coverage_guard"]["max_items"] = 20

    preregistration_path = REPO_ROOT / PREREGISTRATION
    preregistration_sha256 = sha256_file(str(preregistration_path))
    logical_hash = _canonical_sha256(
        {
            "study_id": preregistration["study_id"],
            "candidate": candidate,
            "parent_config_sha256": sha256_file(str(parent_path)),
            "partition_selected_ids_sha256": partition["selected_ids_sha256"],
        }
    )
    config["study"] = {
        "study_id": preregistration["study_id"],
        "family": "cheap_multiroute_followup",
        "variant": candidate["id"],
        "candidate_hash": logical_hash,
        "partition": "dev",
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": preregistration_sha256,
    }

    output_root = REPO_ROOT / OUTPUT_ROOT
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
        "dataset": "GovReport",
        "partition": "dev",
        "partition_rows": len(ordered_ids),
        "partition_manifest_path": spec["manifest"],
        "partition_manifest_sha256": spec["manifest_sha256"],
        "partition_selected_ids_sha256": partition["selected_ids_sha256"],
        "length_policy_path": spec["length_policy"],
        "length_policy_sha256": spec["length_policy_sha256"],
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": preregistration_sha256,
        "family": "cheap_multiroute_followup",
        "candidate": candidate["id"],
        "candidate_hash": logical_hash,
        "declared_delta": candidate["delta"],
        "supersedes_original_failure": False,
        "trigger_candidate": "G11_dataset_structure_guard",
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
        if diagnostics["selector_candidate_size_max"] > 60:
            raise ValueError("section-guard follow-up exceeded total cap 60")
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
            "interpretation": "feasibility follow-up only; original G11 remains failed",
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
            "dataset": "GovReport",
            "partition": "dev",
            "family": "cheap_multiroute_followup",
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
            "reason": "feasibility follow-up only; original G11 remains failed",
            "comparison_family_size": preregistration[
                "multiple_comparison_count_after_followup"
            ],
            "test_split_accessed": False,
        }
    )
    return result


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False))


if __name__ == "__main__":
    main()
