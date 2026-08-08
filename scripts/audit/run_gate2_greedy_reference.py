"""Run preregistered metric-specific greedy references on frozen dev only.

There is no split argument and no route to dev-test or test. Documents are
independent, so process workers are an execution-only optimization; results
are assembled in the exact frozen-manifest order and checkpointed per row.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

from scripts.audit.run_length_contract_study import (
    REPO_ROOT,
    _append_search_log,
    _canonical_sha256,
    _dependency_versions,
    _git_commit,
    _relative,
    _utc_now,
    _write_json,
)
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file
from src.eval.oracle import greedy_reference_run
from src.eval.rouge import DEFAULT_METRICS
from src.utils.io import read_jsonl


PREREGISTRATION = "configs/preregistrations/gate2_greedy_reference_v1.json"
PREREGISTRATION_SHA256 = "04235301d400a1a7bd502879cd8d59b5a5c27cafc902b23ee16bde3435c9f735"
OUTPUT_ROOT = REPO_ROOT / "runs_v2/gate2_greedy_reference_v1"


def _load_protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    actual = sha256_file(str(path))
    if actual != PREREGISTRATION_SHA256:
        raise ValueError(f"greedy-reference preregistration drifted: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_greedy_reference_scores":
        raise ValueError("greedy-reference protocol is not frozen before scores")
    if protocol.get("partition") != "dev":
        raise ValueError("greedy-reference protocol must use frozen dev")
    if protocol.get("dev_test_accessed") is not False:
        raise ValueError("greedy-reference protocol must prohibit dev-test")
    if protocol.get("test_split_prohibited") is not True:
        raise ValueError("greedy-reference protocol must prohibit test")
    if tuple(protocol["configurations"]["optimization_targets"]) != DEFAULT_METRICS:
        raise ValueError("greedy-reference targets differ from evaluator metrics")
    return protocol


def _load_frozen_rows(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = REPO_ROOT / str(spec["input"])
    manifest_path = REPO_ROOT / str(spec["manifest"])
    policy_path = REPO_ROOT / str(spec["length_policy"])
    for path, expected in (
        (input_path, spec["input_sha256"]),
        (manifest_path, spec["manifest_sha256"]),
        (policy_path, spec["length_policy_sha256"]),
    ):
        if sha256_file(str(path)) != expected:
            raise ValueError(f"frozen greedy-reference input drifted: {_relative(path)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("base_split") != "validation":
        raise ValueError("greedy-reference manifest must be based on validation")
    partition = manifest["partitions"]["dev"]
    ordered_ids = [str(value) for value in partition["selected_ids"]]
    if selected_ids_sha256(ordered_ids) != partition["selected_ids_sha256"]:
        raise ValueError("greedy-reference frozen ID digest drifted")
    if len(ordered_ids) != int(spec["partition_rows"]):
        raise ValueError("greedy-reference preregistered row count drifted")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if int(policy["max_words"]) != int(spec["max_words"]):
        raise ValueError("greedy-reference max_words differs from frozen A1 policy")

    by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(str(input_path)):
        row_id = str(row.get("id"))
        if row_id in by_id:
            raise ValueError(f"duplicate canonical row ID: {row_id}")
        by_id[row_id] = dict(row)
    missing = [row_id for row_id in ordered_ids if row_id not in by_id]
    if missing:
        raise ValueError(f"greedy-reference manifest IDs missing from input: {missing[:3]}")
    rows = [by_id[row_id] for row_id in ordered_ids]
    return rows, partition


def _evaluate_row(task: tuple[int, dict[str, Any], str, int]) -> dict[str, Any]:
    position, row, target, max_words = task
    result = greedy_reference_run(
        [row],
        max_words=max_words,
        target_metric=target,
        metrics=DEFAULT_METRICS,
    )
    selection = result["selections"][0]
    return {
        "position": position,
        "id": str(row.get("id")),
        "optimization_target": target,
        "selected_indices": selection["selected_indices"],
        "selected_words": selection["selected_words"],
        "selected_sentences": selection["selected_sentences"],
        "scores": result["scores"],
    }


def _checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in read_jsonl(str(path))]


def _validate_checkpoint_prefix(
    checkpoint: Iterable[Mapping[str, Any]], ordered_ids: list[str], target: str
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in checkpoint]
    if len(rows) > len(ordered_ids):
        raise ValueError("greedy-reference checkpoint is longer than frozen dev")
    ids = [str(row.get("id")) for row in rows]
    if ids != ordered_ids[: len(ids)]:
        raise ValueError("greedy-reference checkpoint is not the exact frozen ID prefix")
    for position, row in enumerate(rows):
        if row.get("position") != position:
            raise ValueError("greedy-reference checkpoint positions are not canonical")
        if row.get("optimization_target") != target:
            raise ValueError("greedy-reference checkpoint target drifted")
        if set(row.get("scores", {})) != set(DEFAULT_METRICS):
            raise ValueError("greedy-reference checkpoint metrics are incomplete")
    return rows


def _selected_indices_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    return _canonical_sha256(
        [
            {"id": str(row["id"]), "selected_indices": list(row["selected_indices"])}
            for row in rows
        ]
    )


def _log_interruption(
    *, dataset_label: str, target: str, config_hash: str, rows_path: Path, rows: int
) -> None:
    attempt_root = rows_path.parent / "attempts"
    attempt_number = len(list(attempt_root.glob("attempt_*_interrupted"))) + 1
    attempt_id = f"attempt_{attempt_number:02d}_interrupted"
    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "interrupted",
        "reason": "incomplete row checkpoint detected on explicit resume",
        "dataset": dataset_label,
        "partition": "dev",
        "optimization_target": target,
        "completed_prefix_rows": rows,
        "checkpoint_path": _relative(rows_path),
        "checkpoint_sha256": sha256_file(str(rows_path)),
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    evidence_path = attempt_root / attempt_id / "interruption_evidence.json"
    _write_json(evidence_path, evidence)
    _append_search_log(
        {
            "logged_at_utc": _utc_now(),
            "study_id": "gate2-greedy-reference-v1",
            "dataset": dataset_label,
            "partition": "dev",
            "family": "greedy_reference",
            "candidate": target,
            "method": "metric_specific_greedy_reference",
            "config_hash": config_hash,
            "run_attempt": attempt_id,
            "dev_score": None,
            "dev_test_score": None,
            "status": "failed",
            "promoted": False,
            "reason": evidence["reason"],
            "comparison_family_size": 6,
            "dev_test_accessed": False,
            "test_split_accessed": False,
        }
    )


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "rows": count,
        "rouge": {
            metric: sum(float(row["scores"][metric]) for row in rows) / count
            for metric in DEFAULT_METRICS
        },
        "mean_selected_words": sum(int(row["selected_words"]) for row in rows) / count,
        "mean_selected_sentences": sum(int(row["selected_sentences"]) for row in rows) / count,
        "min_selected_words": min(int(row["selected_words"]) for row in rows),
        "max_selected_words": max(int(row["selected_words"]) for row in rows),
    }


def run_configuration(
    dataset: str, target: str, *, workers: int, resume: bool = False
) -> dict[str, Any]:
    protocol = _load_protocol()
    spec = protocol["datasets"][dataset]
    frozen_rows, partition = _load_frozen_rows(spec)
    ordered_ids = [str(row["id"]) for row in frozen_rows]
    config = {
        "study_id": "gate2-greedy-reference-v1",
        "dataset": dataset,
        "dataset_label": spec["label"],
        "partition": "dev",
        "optimization_target": target,
        "max_words": int(spec["max_words"]),
        "max_sentences": None,
        "algorithm": protocol["configurations"]["search"],
        "report_metrics": list(DEFAULT_METRICS),
        "requested_min_rule": protocol["configurations"]["requested_min_rule"],
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "partition_selected_ids_sha256": partition["selected_ids_sha256"],
    }
    config_hash = _canonical_sha256(config)
    root = OUTPUT_ROOT / dataset / "dev" / target
    rows_path = root / "rows.jsonl"
    evidence_path = root / "evidence.json"
    if evidence_path.exists():
        if resume:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if (
                evidence.get("status") != "completed"
                or evidence.get("scientific_config_sha256") != config_hash
            ):
                raise ValueError("completed greedy-reference evidence drifted")
            return evidence
        raise ValueError(f"refusing to overwrite completed greedy-reference run: {root}")
    if root.exists() and not resume:
        raise ValueError(f"refusing to overwrite partial greedy-reference run: {root}")
    root.mkdir(parents=True, exist_ok=resume)
    config_path = root / "resolved_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("greedy-reference resume config drifted")
    else:
        _write_json(config_path, config)
    config_artifact_sha256 = sha256_file(str(config_path))

    completed = _validate_checkpoint_prefix(
        _checkpoint_rows(rows_path), ordered_ids, target
    )
    if completed and not resume:
        raise ValueError("partial greedy-reference checkpoint requires --resume")
    if completed and len(completed) < len(frozen_rows):
        _log_interruption(
            dataset_label=str(spec["label"]),
            target=target,
            config_hash=config_hash,
            rows_path=rows_path,
            rows=len(completed),
        )

    remaining = [
        (position, frozen_rows[position], target, int(spec["max_words"]))
        for position in range(len(completed), len(frozen_rows))
    ]
    invocation_started_at = _utc_now()
    invocation_started = time.perf_counter()
    try:
        with rows_path.open("a", encoding="utf-8", newline="\n") as stream:
            if remaining:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for row_result in executor.map(_evaluate_row, remaining, chunksize=1):
                        stream.write(json.dumps(row_result, ensure_ascii=False) + "\n")
                        stream.flush()
                        completed.append(row_result)
    except Exception as error:
        failure = {
            "evidence_schema_version": "1.0",
            "measured_at_utc": _utc_now(),
            "status": "failed",
            "failure": f"{type(error).__name__}: {error}",
            "dataset": spec["label"],
            "partition": "dev",
            "optimization_target": target,
            "completed_prefix_rows": len(completed),
            "scientific_config_sha256": config_hash,
            "dev_test_accessed": False,
            "test_split_accessed": False,
        }
        _write_json(root / "failure_evidence.json", failure)
        _append_search_log(
            {
                "logged_at_utc": _utc_now(),
                "study_id": "gate2-greedy-reference-v1",
                "dataset": spec["label"],
                "partition": "dev",
                "family": "greedy_reference",
                "candidate": target,
                "method": "metric_specific_greedy_reference",
                "config_hash": config_hash,
                "run_attempt": "failed_execution",
                "dev_score": None,
                "dev_test_score": None,
                "status": "failed",
                "promoted": False,
                "reason": failure["failure"],
                "comparison_family_size": 6,
                "dev_test_accessed": False,
                "test_split_accessed": False,
            }
        )
        raise

    completed = _validate_checkpoint_prefix(completed, ordered_ids, target)
    if len(completed) != len(frozen_rows):
        raise ValueError("greedy-reference run ended without full frozen-dev coverage")
    metrics = _aggregate(completed)
    _write_json(root / "metrics.json", metrics)
    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "invocation_started_at_utc": invocation_started_at,
        "status": "completed",
        "study_id": "gate2-greedy-reference-v1",
        "implementation_commit": _git_commit(),
        "dataset": spec["label"],
        "partition": "dev",
        "partition_rows": len(frozen_rows),
        "optimization_target": target,
        "method": "metric_specific_greedy_reference",
        "exact_upper_bound": False,
        "max_words": int(spec["max_words"]),
        "requested_min_forced": False,
        "config_path": _relative(config_path),
        "scientific_config_sha256": config_hash,
        "config_artifact_sha256": config_artifact_sha256,
        "preregistration_path": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "input_path": spec["input"],
        "input_sha256": spec["input_sha256"],
        "partition_manifest_path": spec["manifest"],
        "partition_manifest_sha256": spec["manifest_sha256"],
        "partition_selected_ids_sha256": partition["selected_ids_sha256"],
        "rows_path": _relative(rows_path),
        "rows_sha256": sha256_file(str(rows_path)),
        "selected_indices_sha256": _selected_indices_digest(completed),
        "dependency_versions": _dependency_versions(),
        "execution": {
            "workers": workers,
            "ordered_result_assembly": True,
            "invocation_wall_seconds": time.perf_counter() - invocation_started,
        },
        "metrics": metrics,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(evidence_path, evidence)
    _append_search_log(
        {
            "logged_at_utc": _utc_now(),
            "study_id": "gate2-greedy-reference-v1",
            "dataset": spec["label"],
            "partition": "dev",
            "family": "greedy_reference",
            "candidate": target,
            "method": "metric_specific_greedy_reference",
            "config_path": _relative(config_path),
            "config_hash": config_hash,
            "run_attempt": "final",
            "dev_score": metrics["rouge"][target],
            "dev_test_score": None,
            "status": "completed",
            "promoted": False,
            "reason": "headroom diagnostic only; not a promotion candidate",
            "comparison_family_size": 6,
            "dev_test_accessed": False,
            "test_split_accessed": False,
        }
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("multinews", "govreport"))
    parser.add_argument("--target", required=True, choices=DEFAULT_METRICS)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    evidence = run_configuration(
        args.dataset, args.target, workers=args.workers, resume=args.resume
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
