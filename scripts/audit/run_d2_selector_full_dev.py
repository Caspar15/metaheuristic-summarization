"""Run the preregistered full-dev S02b selector screen.

The CLI deliberately exposes no split.  It can only resolve the frozen dev
membership for Multi-News or GovReport, and it never contains a test path.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

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


PREREGISTRATION = Path("configs/preregistrations/d2_selector_full_dev_v1.json")
PREREGISTRATION_SHA256 = (
    "98d0093a7acb22cfbe43aa35bd5aed33fb7fbab85639ac1c920ee15ba8b11382"
)
DATASET_LABELS = {"multinews": "Multi-News", "govreport": "GovReport"}


def _load_preregistration() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    if sha256_file(str(path)) != PREREGISTRATION_SHA256:
        raise ValueError("frozen D2 selector preregistration SHA-256 drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("partition") != "dev" or value.get("dev_test_access") != "none":
        raise ValueError("D2 selector screen must remain bound to frozen dev")
    if not value.get("test_split_prohibited"):
        raise ValueError("D2 selector screen must prohibit test")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 14:
        raise ValueError("D2 selector screen must contain exactly 14 candidates")
    ids = [candidate.get("id") for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("D2 selector candidate IDs must be unique")
    if sum(bool(candidate.get("reuse_base_run")) for candidate in candidates) != 1:
        raise ValueError("D2 selector screen must declare exactly one reused anchor")
    return value


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: dict[str, Any] = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"D2 delta parent {part!r} is absent for {dotted!r}")
        node = child
    node[parts[-1]] = copy.deepcopy(value)


def _validate_base(config: Mapping[str, Any]) -> None:
    if config["compute_budget"]["enabled_routes"] != [
        "lexical",
        "semantic",
        "graph",
    ]:
        raise ValueError("D2 base must be the three-route S02b config")
    actual_budget = {
        key: int(config["candidate_budget"][key])
        for key in ("route_top_k", "min_per_route", "total")
    }
    if actual_budget != {"route_top_k": 40, "min_per_route": 20, "total": 80}:
        raise ValueError(f"D2 S02b candidate budget drift: {actual_budget!r}")
    if int(config["coverage_guard"].get("max_items", -1)) != 20:
        raise ValueError("D2 S02b guard cap drift")
    if config["selector"].get("salience_source") != "rrf_fusion":
        raise ValueError("D2 S02b salience source drift")
    if config["selector"].get("similarity_source") != "pipeline_similarity":
        raise ValueError("D2 anchor must start from TF-IDF pipeline similarity")
    if config["optimizer"].get("method") != "greedy":
        raise ValueError("D2 anchor must start from Greedy")


def resolve_candidate(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    method = str(candidate.get("method", "")).lower()
    if method not in {"greedy", "mmr", "nsga2"}:
        raise ValueError(f"unsupported D2 selector {method!r}")
    config["optimizer"]["method"] = method
    for dotted, value in dict(candidate.get("delta", {})).items():
        _set_dotted(config, dotted, value)
    if config["selector"].get("similarity_source") not in {
        "pipeline_similarity",
        "sbert",
    }:
        raise ValueError("D2 selector similarity must be TF-IDF or pinned SBERT")
    if method == "mmr":
        weight = float(config["optimizer"].get("lambda_relevance", -1))
        if weight not in {0.1, 0.3, 0.5, 0.7, 0.9}:
            raise ValueError("D2 MMR lambda drifted outside the frozen grid")
    if method == "nsga2" and (
        int(config["optimizer"].get("pop_size", -1)) != 64
        or int(config["optimizer"].get("n_gen", -1)) != 80
    ):
        raise ValueError("D2 NSGA-II must use the frozen 64x80 budget")
    return config


def _archive_incomplete(candidate_root: Path, method: str) -> None:
    run_path = candidate_root / method / "run"
    if not run_path.exists():
        return
    attempts = candidate_root / method / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    number = 1
    while (attempts / f"attempt_{number:02d}_interrupted").exists():
        number += 1
    destination = attempts / f"attempt_{number:02d}_interrupted"
    shutil.move(str(run_path), str(destination))
    _write_json(
        destination / "interruption_evidence.json",
        {
            "evidence_schema_version": "1.0",
            "measured_at_utc": _utc_now(),
            "status": "failed",
            "failure_type": "external_runner_interruption",
            "dev_test_accessed": False,
            "test_split_accessed": False,
            "archived_run_path": _relative(destination),
        },
    )


def _load_anchor(dataset_registration: Mapping[str, Any], expected_rows: int) -> dict[str, Any]:
    run_root = REPO_ROOT / str(dataset_registration["base_run"])
    metrics_path = run_root / "metrics.json"
    evidence_path = run_root / "evidence.json"
    per_example_path = run_root / "per_example.jsonl"
    for path in (metrics_path, evidence_path, per_example_path):
        if not path.is_file():
            raise ValueError(f"D2 anchor artifact is missing: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if int(metrics.get("rows", -1)) != expected_rows:
        raise ValueError("D2 anchor row count drift")
    if evidence.get("test_split_accessed") is not False:
        raise ValueError("D2 anchor lacks a false test guard")
    return {
        "status": "completed_reused_anchor",
        "candidate": "S00_greedy_tfidf_anchor",
        "method": "greedy",
        "metrics": metrics,
        "run_path": _relative(run_root),
        "per_example_path": _relative(per_example_path),
        "evidence_sha256": sha256_file(str(evidence_path)),
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }


def run(dataset: str, *, resume: bool = False, only: str | None = None) -> dict[str, Any]:
    if dataset not in DATASET_LABELS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    label = DATASET_LABELS[dataset]
    preregistration = _load_preregistration()
    registration = preregistration["datasets"][label]
    base_path = REPO_ROOT / str(registration["base_config"])
    if sha256_file(str(base_path)) != registration["base_config_sha256"]:
        raise ValueError(f"{label} D2 base-config SHA-256 drift")
    base = load_yaml(str(base_path))
    _validate_base(base)

    spec = _load_study(dataset)
    partition = spec["manifest_object"]["partitions"]["dev"]
    ordered_ids = partition["selected_ids"]
    if len(ordered_ids) != int(registration["rows"]):
        raise ValueError(f"{label} D2 row-count drift")
    if selected_ids_sha256(ordered_ids) != partition["selected_ids_sha256"]:
        raise ValueError(f"{label} D2 frozen-ID digest drift")
    input_path = REPO_ROOT / spec["input"]
    gold = _load_gold(input_path, ordered_ids)

    candidates = preregistration["candidates"]
    candidate_ids = {str(candidate["id"]) for candidate in candidates}
    if only is not None and only not in candidate_ids:
        raise ValueError(f"unknown D2 candidate {only!r}")
    output_root = REPO_ROOT / "runs_v2" / "d2_selector_full_dev_v1" / dataset / "dev"
    if output_root.exists() and not resume:
        raise ValueError(f"refusing to overwrite D2 study: {output_root}")
    output_root.mkdir(parents=True, exist_ok=resume)

    cache_root = (
        REPO_ROOT
        / "runs_v2"
        / "gate2_baseline_matrix_v1"
        / dataset
        / "dev"
        / "plm"
        / "_embedding_cache"
    )
    if not cache_root.is_dir():
        raise ValueError(f"F-51-audited D2 cache root is missing: {cache_root}")
    prereg_sha = sha256_file(str(REPO_ROOT / PREREGISTRATION))
    results: dict[str, Any] = {}

    for candidate in candidates:
        candidate_id = str(candidate["id"])
        if only is not None and candidate_id != only:
            continue
        if candidate.get("reuse_base_run"):
            results[candidate_id] = _load_anchor(registration, len(ordered_ids))
            continue
        method = str(candidate["method"])
        config = resolve_candidate(base, candidate)
        logical_hash = _canonical_sha256(
            {
                "study_id": preregistration["study_id"],
                "dataset": label,
                "candidate": candidate,
                "base_config_sha256": registration["base_config_sha256"],
                "partition_selected_ids_sha256": partition["selected_ids_sha256"],
            }
        )
        config["study"] = {
            "study_id": preregistration["study_id"],
            "family": "selector_full_dev",
            "variant": candidate_id,
            "candidate_hash": logical_hash,
            "partition": "dev",
            "preregistration_path": PREREGISTRATION.as_posix(),
            "preregistration_sha256": prereg_sha,
        }
        candidate_root = output_root / candidate_id
        summary_path = candidate_root / "candidate_summary.json"
        config_path = candidate_root / "resolved_config.yaml"
        if summary_path.is_file():
            if not resume:
                raise ValueError(f"D2 candidate already exists: {candidate_root}")
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
            if prior.get("candidate_hash") != logical_hash:
                raise ValueError(f"D2 resume hash drift for {candidate_id}")
            results[candidate_id] = prior
            continue
        if candidate_root.exists():
            if not resume:
                raise ValueError(f"D2 candidate directory already exists: {candidate_root}")
            _archive_incomplete(candidate_root, method)
        else:
            candidate_root.mkdir(parents=True, exist_ok=False)
        config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        if config_path.is_file():
            if config_path.read_text(encoding="utf-8") != config_text:
                raise ValueError(f"D2 resume config drift for {candidate_id}")
        else:
            config_path.write_text(config_text, encoding="utf-8", newline="\n")
        config_sha = sha256_file(str(config_path))
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
            "preregistration_sha256": prereg_sha,
            "candidate": candidate_id,
            "candidate_hash": logical_hash,
            "declared_delta": candidate["delta"],
            "execution_optimization": {
                "embedding_cache": {
                    "enabled": True,
                    "root": _relative(cache_root),
                    "scientific_config_changed": False,
                }
            },
            "dev_test_accessed": False,
        }
        try:
            metrics, _ = _run_method(
                method,
                config_path=config_path,
                config_sha256=config_sha,
                input_path=input_path,
                candidate_root=candidate_root,
                ordered_ids=ordered_ids,
                gold=gold,
                study_context=context,
                subprocess_env={"META_SUM_EMBEDDING_CACHE_DIR": str(cache_root)},
            )
            diagnostics = _candidate_diagnostics(
                candidate_root / method / "run" / "predictions.jsonl"
            )
            result = {
                "evidence_schema_version": "1.0",
                "measured_at_utc": _utc_now(),
                "status": "completed",
                "test_split_accessed": False,
                **context,
                "method": method,
                "config_path": _relative(config_path),
                "config_sha256": config_sha,
                "metrics": metrics,
                "candidate_diagnostics": diagnostics,
            }
        except Exception as error:
            result = {
                "evidence_schema_version": "1.0",
                "measured_at_utc": _utc_now(),
                "status": "failed",
                "test_split_accessed": False,
                **context,
                "method": method,
                "config_path": _relative(config_path),
                "config_sha256": config_sha,
                "failure": f"{type(error).__name__}: {error}",
            }
        _write_json(summary_path, result)
        _append_search_log(
            {
                "logged_at_utc": _utc_now(),
                "study_id": preregistration["study_id"],
                "dataset": label,
                "partition": "dev",
                "family": "selector_full_dev",
                "candidate": candidate_id,
                "candidate_hash": logical_hash,
                "config_path": _relative(config_path),
                "config_hash": config_sha,
                "run_attempt": "final",
                "dev_score": (
                    float(result["metrics"]["macro_rouge"])
                    if result["status"] == "completed"
                    else None
                ),
                "dev_test_score": None,
                "status": result["status"],
                "promoted": None,
                "reason": "full-dev matched selector screen; pending paired analysis",
                "comparison_family_size": 57,
                "test_split_accessed": False,
            }
        )
        results[candidate_id] = result

    study_summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": (
            "completed"
            if only is None
            and len(results) == 14
            and all(str(row["status"]).startswith("completed") for row in results.values())
            else "partial"
        ),
        "study_id": preregistration["study_id"],
        "dataset": label,
        "partition": "dev",
        "partition_rows": len(ordered_ids),
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": prereg_sha,
        "candidates": results,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(output_root / "study_summary.json", study_summary)
    return study_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_LABELS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.dataset, resume=args.resume, only=args.candidate),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
