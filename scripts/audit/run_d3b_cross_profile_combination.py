"""Run the single preregistered D3b combination per task profile on dev."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.audit.run_d3a_router_fusion_full_dev import (
    _run_candidate,
    _selected_rows,
    resolve_candidate as validate_resolved_candidate,
)
from scripts.audit.run_greedy_sensitivity import REPO_ROOT, _load_study
from scripts.audit.run_length_contract_study import (
    _append_search_log,
    _canonical_sha256,
    _dependency_versions,
    _git_commit,
    _load_gold,
    _relative,
    _utc_now,
    _write_json,
)
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file, validate_dataset_policy_request
from src.utils.io import load_yaml


PREREGISTRATION = Path("configs/preregistrations/d3b_cross_profile_combination_v1.json")
PREREGISTRATION_SHA256 = (
    "58c6c98c1953416463bf000c102d351a320ef3634be91d3fd478827ac12bc112"
)
DATASET_LABELS = {"multinews": "Multi-News", "govreport": "GovReport"}


def _protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    if sha256_file(str(path)) != PREREGISTRATION_SHA256:
        raise ValueError("D3b preregistration SHA-256 drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("partition") != "dev" or value.get("dev_test_access") != "none":
        raise ValueError("D3b must remain on frozen dev")
    if value.get("test_split_prohibited") is not True:
        raise ValueError("D3b must prohibit test")
    source = REPO_ROOT / value["source_analysis"]
    if sha256_file(str(source)) != value["source_analysis_sha256"]:
        raise ValueError("D3b source analysis drifted")
    if value.get("scores_observed_after_registration") is not False:
        raise ValueError("D3b registration must precede combination scores")
    return value


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: dict[str, Any] = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"D3b delta parent {part!r} is absent")
        node = child
    node[parts[-1]] = copy.deepcopy(value)


def resolve_combination(
    base: Mapping[str, Any], registration: Mapping[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    for dotted, value in registration["combination_delta"].items():
        _set_dotted(config, dotted, value)
    validate_resolved_candidate(config, {"delta": {}})
    if config["features"]["tf_isf"].get("use_bigrams") is not True:
        raise ValueError("D3b combination must enable bigrams")
    if float(config["features"]["weights"].get("position", -1)) != 0.2:
        raise ValueError("D3b combination must use frozen position weight")
    weights = config["candidates"].get("route_weights")
    if not isinstance(weights, dict) or len(weights) != 1:
        raise ValueError("D3b combination must change exactly one route weight")
    return config


def run(dataset: str, *, workers: int, resume: bool = False) -> dict[str, Any]:
    if dataset not in DATASET_LABELS:
        raise ValueError(f"unsupported D3b dataset {dataset!r}")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    label = DATASET_LABELS[dataset]
    protocol = _protocol()
    registration = protocol["datasets"][label]
    spec = _load_study(dataset)
    input_path = REPO_ROOT / spec["input"]
    base_path = REPO_ROOT / registration["base_config"]
    if sha256_file(str(base_path)) != registration["base_config_sha256"]:
        raise ValueError("D3b base config drifted")
    base = load_yaml(str(base_path))
    validate_dataset_policy_request(base, str(input_path), "validation")
    partition = spec["manifest_object"]["partitions"]["dev"]
    ordered_ids = [str(value) for value in partition["selected_ids"]]
    if len(ordered_ids) != int(registration["rows"]):
        raise ValueError("D3b full-dev row count drifted")
    if selected_ids_sha256(ordered_ids) != partition["selected_ids_sha256"]:
        raise ValueError("D3b full-dev ID digest drifted")
    gold = _load_gold(input_path, ordered_ids)
    anchor_summary_path = REPO_ROOT / registration["anchor_summary"]
    anchor = json.loads(anchor_summary_path.read_text(encoding="utf-8"))
    if anchor.get("status") != "completed":
        raise ValueError("D3b D3a anchor is incomplete")
    if anchor.get("dev_test_accessed") is not False or anchor.get("test_split_accessed") is not False:
        raise ValueError("D3b anchor lacks protected-split guards")

    output_root = REPO_ROOT / "runs_v2/d3b_cross_profile_combination_v1" / dataset / "dev"
    if output_root.exists() and not resume:
        raise ValueError(f"refusing to overwrite D3b study: {output_root}")
    output_root.mkdir(parents=True, exist_ok=resume)
    candidate_id = protocol["candidate"]["id"]
    candidate_root = output_root / candidate_id
    summary_path = candidate_root / "candidate_summary.json"
    logical_hash = _canonical_sha256({
        "study_id": protocol["study_id"],
        "dataset": label,
        "candidate": protocol["candidate"],
        "combination_delta": registration["combination_delta"],
        "base_config_sha256": registration["base_config_sha256"],
        "selected_ids_sha256": partition["selected_ids_sha256"],
    })
    if summary_path.is_file():
        if not resume:
            raise ValueError("D3b candidate already exists")
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("candidate_hash") != logical_hash:
            raise ValueError("D3b resume hash drifted")
        if prior.get("status") != "completed":
            raise ValueError("failed D3b attempt is preserved; archive before retry")
        result = prior
    else:
        candidate_root.mkdir(parents=True, exist_ok=False)
        config = resolve_combination(base, registration)
        config["study"] = {
            "study_id": protocol["study_id"],
            "family": protocol["candidate"]["family"],
            "variant": candidate_id,
            "candidate_hash": logical_hash,
            "partition": "dev",
            "preregistration_path": PREREGISTRATION.as_posix(),
            "preregistration_sha256": PREREGISTRATION_SHA256,
        }
        config_path = candidate_root / "resolved_config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8", newline="\n"
        )
        cache_root = REPO_ROOT / "runs_v2/gate2_baseline_matrix_v1" / dataset / "dev/plm/_embedding_cache"
        if not cache_root.is_dir():
            raise ValueError("D3b embedding cache is missing")
        os.environ["META_SUM_EMBEDDING_CACHE_DIR"] = str(cache_root)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        started_at = _utc_now()
        started = time.perf_counter()
        try:
            metrics, artifacts = _run_candidate(
                rows=_selected_rows(input_path, ordered_ids),
                ordered_ids=ordered_ids,
                gold=gold,
                config=config,
                candidate_root=candidate_root,
                workers=workers,
            )
            result = {
                "evidence_schema_version": "1.0",
                "measured_at_utc": _utc_now(),
                "started_at_utc": started_at,
                "status": "completed",
                "study_id": protocol["study_id"],
                "dataset": label,
                "partition": "dev",
                "candidate": candidate_id,
                "candidate_hash": logical_hash,
                "declared_delta": registration["combination_delta"],
                "config_path": _relative(config_path),
                "config_sha256": sha256_file(str(config_path)),
                "implementation_commit": _git_commit(),
                "dependency_versions": _dependency_versions(),
                "execution": {
                    "workers": workers,
                    "threads_per_worker": 1,
                    "streamed_predictions": True,
                    "elapsed_seconds": time.perf_counter() - started,
                    "embedding_cache_root": _relative(cache_root),
                },
                "anchor_candidate": registration["anchor_candidate"],
                "anchor_summary_path": _relative(anchor_summary_path),
                "anchor_summary_sha256": sha256_file(str(anchor_summary_path)),
                "metrics": metrics,
                **artifacts,
                "dev_test_accessed": False,
                "test_split_accessed": False,
            }
        except Exception as error:
            result = {
                "evidence_schema_version": "1.0",
                "measured_at_utc": _utc_now(),
                "started_at_utc": started_at,
                "status": "failed",
                "study_id": protocol["study_id"],
                "dataset": label,
                "partition": "dev",
                "candidate": candidate_id,
                "candidate_hash": logical_hash,
                "failure": f"{type(error).__name__}: {error}",
                "dev_test_accessed": False,
                "test_split_accessed": False,
            }
        _write_json(summary_path, result)
        _append_search_log({
            "logged_at_utc": _utc_now(),
            "study_id": protocol["study_id"],
            "dataset": label,
            "partition": "dev",
            "family": protocol["candidate"]["family"],
            "candidate": candidate_id,
            "candidate_hash": logical_hash,
            "config_path": _relative(config_path),
            "config_hash": sha256_file(str(config_path)),
            "dev_score": result.get("metrics", {}).get("macro_rouge"),
            "dev_test_score": None,
            "status": result["status"],
            "promoted": None,
            "reason": "final D3b combination; formal 100k paired gate pending",
            "comparison_family_size": 85,
            "test_split_accessed": False,
        })
    summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed" if result["status"] == "completed" else "failed",
        "study_id": protocol["study_id"],
        "dataset": label,
        "partition": "dev",
        "partition_rows": len(ordered_ids),
        "anchor": {
            "candidate": registration["anchor_candidate"],
            "macro": anchor["metrics"]["macro_rouge"],
            "summary_path": _relative(anchor_summary_path),
        },
        "combination": result,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(output_root / "study_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LABELS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.dataset, workers=args.workers, resume=args.resume)
    print(json.dumps({"status": result["status"], "dataset": result["dataset"]}))


if __name__ == "__main__":
    main()
