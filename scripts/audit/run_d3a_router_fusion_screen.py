"""Run the preregistered D3a router/fusion screen on frozen dev pilots only.

The CLI has no dev-test or test option.  It validates each full validation
input against the frozen data policy, then scores only reference-blind pilot
IDs.  Row workers are bounded so ``--workers 16`` does not eagerly materialize
the entire study or launch 16 whole-dataset model processes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import yaml

from scripts.audit.freeze_selector_pilot import file_sha256
from scripts.audit.run_greedy_sensitivity import REPO_ROOT, _load_study
from scripts.audit.run_length_contract_study import (
    _append_search_log,
    _canonical_sha256,
    _dependency_versions,
    _git_commit,
    _relative,
    _selected_indices_digest,
    _utc_now,
    _write_json,
    _write_jsonl,
)
from scripts.audit.run_selector_comparison import load_frozen_rows
from src.data.policy import sha256_file, validate_dataset_policy_request, verify_pin
from src.data.schemas import extract_references
from src.eval.rouge import DEFAULT_METRICS, rouge_scores
from src.pipeline.select_sentences import summarize_one, validate_requested_split
from src.utils.io import load_yaml


PREREGISTRATION = Path("configs/preregistrations/d3a_router_fusion_screen_v1.json")
PREREGISTRATION_SHA256 = (
    "07de205a39b781389573b0c44fa526474f2226ca329b52a027e0288c770de58a"
)
DATASET_LABELS = {"multinews": "Multi-News", "govreport": "GovReport"}


def _load_preregistration() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    if sha256_file(str(path)) != PREREGISTRATION_SHA256:
        raise ValueError("frozen D3a preregistration SHA-256 drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("dev_test_access") != "none":
        raise ValueError("D3a must not access dev-test")
    if value.get("test_split_prohibited") is not True:
        raise ValueError("D3a must prohibit test access")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 14:
        raise ValueError("D3a must contain exactly 14 candidates")
    ids = [str(row.get("id")) for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("D3a candidate IDs must be unique")
    if ids[0] != "R00_anchor" or candidates[0].get("delta") != {}:
        raise ValueError("D3a first candidate must be the unchanged anchor")
    return value


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: dict[str, Any] = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"D3a delta parent {part!r} is absent for {dotted!r}")
        node = child
    node[parts[-1]] = copy.deepcopy(value)


def resolve_candidate(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    for dotted, value in dict(candidate.get("delta", {})).items():
        _set_dotted(config, dotted, value)
    if config["compute_budget"]["enabled_routes"] != [
        "lexical", "semantic", "graph"
    ]:
        raise ValueError("D3a must retain the three frozen routes")
    if config["selector"].get("salience_source") != "rrf_fusion":
        raise ValueError("D3a must retain provenance salience")
    if config["selector"].get("similarity_source") != "pipeline_similarity":
        raise ValueError("D3a must retain D2 TF-IDF selector similarity")
    if int(config["candidate_budget"]["route_top_k"]) > 120:
        raise ValueError("D3a route_top_k exceeds preregistered cost bound")
    if int(config["candidate_budget"]["total"]) > 160:
        raise ValueError("D3a total exceeds preregistered cost bound")
    weights = config.get("candidates", {}).get("route_weights", {})
    if weights and (not isinstance(weights, dict) or len(weights) != 1):
        raise ValueError("D3a weighted variants must change exactly one route")
    return config


def _worker(task: tuple[int, dict[str, Any], dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    position, row, config = task
    prediction = summarize_one(row, config)
    return position, prediction


def _bounded_rows(
    rows: list[dict[str, Any]], config: dict[str, Any], *, workers: int
) -> Iterator[dict[str, Any]]:
    if not 1 <= workers <= 16:
        raise ValueError("D3a workers must be in [1, 16]")
    tasks: Iterable[tuple[int, dict[str, Any], dict[str, Any]]] = (
        (position, row, config) for position, row in enumerate(rows)
    )
    iterator = iter(tasks)
    pending = {}
    buffered: dict[int, dict[str, Any]] = {}
    next_position = 0
    exhausted = False
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while pending or not exhausted:
            while not exhausted and len(pending) < workers:
                try:
                    task = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(_worker, task)
                pending[future] = task[0]
            if not pending:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                expected = pending.pop(future)
                position, prediction = future.result()
                if position != expected:
                    raise RuntimeError("D3a worker returned the wrong row position")
                buffered[position] = prediction
            while next_position in buffered:
                yield buffered.pop(next_position)
                next_position += 1


def _candidate_metrics(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    means, per_example = rouge_scores(
        [str(row.get("summary", "")) for row in predictions],
        [extract_references(row) for row in rows],
        metrics=DEFAULT_METRICS,
        return_per_example=True,
    )
    per_example_rows = [
        {"id": row["id"], **scores}
        for row, scores in zip(rows, per_example)
    ]
    lengths = [len(str(row.get("summary", "")).split()) for row in predictions]
    feasible = [bool(row.get("feasible")) for row in predictions]
    metrics = {
        "evaluation_protocol": "multisentence_lsum",
        "rows": len(rows),
        "rouge": means,
        "macro_rouge": float(np.mean([means[name] for name in DEFAULT_METRICS])),
        "feasible_rows": int(sum(feasible)),
        "infeasible_rows": int(len(feasible) - sum(feasible)),
        "summary_words": {
            "mean": float(np.mean(lengths)),
            "min": int(min(lengths)),
            "max": int(max(lengths)),
        },
    }
    return metrics, per_example_rows


def run(dataset: str, *, workers: int, resume: bool = False) -> dict[str, Any]:
    if dataset not in DATASET_LABELS:
        raise ValueError(f"unsupported D3a dataset {dataset!r}")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    label = DATASET_LABELS[dataset]
    prereg = _load_preregistration()
    registration = prereg["datasets"][label]
    spec = _load_study(dataset)
    input_path = REPO_ROOT / spec["input"]
    base_path = REPO_ROOT / registration["base_config"]
    manifest_path = REPO_ROOT / registration["pilot_manifest"]
    if sha256_file(str(base_path)) != registration["base_config_sha256"]:
        raise ValueError("D3a base config SHA-256 drifted")
    pin_status = verify_pin(str(manifest_path), registration["pilot_manifest_sha256"])
    if pin_status == "legacy":
        print(f"[legacy pin] {manifest_path} (CRLF-era pin, see errata)")
    base = load_yaml(str(base_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_preflight = validate_dataset_policy_request(
        base, str(input_path), "validation"
    )
    rows = load_frozen_rows(input_path, manifest)
    if len(rows) != int(registration["pilot_rows"]):
        raise ValueError("D3a pilot row count drifted")
    for row in rows:
        validate_requested_split(row, "validation")

    output_root = REPO_ROOT / "runs_v2" / "d3a_router_fusion_screen_v1" / dataset / "pilot"
    if output_root.exists() and not resume:
        raise ValueError(f"refusing to overwrite D3a study: {output_root}")
    output_root.mkdir(parents=True, exist_ok=resume)
    cache_root = REPO_ROOT / "runs_v2" / "gate2_baseline_matrix_v1" / dataset / "dev" / "plm" / "_embedding_cache"
    if not cache_root.is_dir():
        raise ValueError(f"D3a audited embedding cache is missing: {cache_root}")
    os.environ["META_SUM_EMBEDDING_CACHE_DIR"] = str(cache_root)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    prereg_sha = sha256_file(str(REPO_ROOT / PREREGISTRATION))
    results: dict[str, Any] = {}
    for candidate in prereg["candidates"]:
        candidate_id = str(candidate["id"])
        candidate_root = output_root / candidate_id
        summary_path = candidate_root / "candidate_summary.json"
        logical_hash = _canonical_sha256({
            "study_id": prereg["study_id"],
            "dataset": label,
            "candidate": candidate,
            "base_config_sha256": registration["base_config_sha256"],
            "pilot_ids_sha256": manifest["selected_ids_sha256"],
        })
        if summary_path.is_file():
            if not resume:
                raise ValueError(f"D3a candidate already exists: {candidate_root}")
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
            if prior.get("candidate_hash") != logical_hash:
                raise ValueError(f"D3a resume hash drift for {candidate_id}")
            if prior.get("status") == "completed":
                results[candidate_id] = prior
                continue
            raise ValueError(
                f"D3a failed candidate {candidate_id} is preserved; archive it before retry"
            )
        candidate_root.mkdir(parents=True, exist_ok=False)
        config = resolve_candidate(base, candidate)
        config["study"] = {
            "study_id": prereg["study_id"],
            "family": candidate["family"],
            "variant": candidate_id,
            "candidate_hash": logical_hash,
            "partition": "dev_pilot",
            "preregistration_path": PREREGISTRATION.as_posix(),
            "preregistration_sha256": prereg_sha,
        }
        config_path = candidate_root / "resolved_config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8", newline="\n"
        )
        started_at = _utc_now()
        started = time.perf_counter()
        try:
            predictions = list(_bounded_rows(rows, config, workers=workers))
            elapsed = time.perf_counter() - started
            if [row["id"] for row in predictions] != [row["id"] for row in rows]:
                raise RuntimeError("D3a prediction order differs from frozen pilot")
            predictions_path = candidate_root / "predictions.jsonl"
            _write_jsonl(predictions_path, predictions)
            metrics, per_example = _candidate_metrics(rows, predictions)
            _write_json(candidate_root / "metrics.json", metrics)
            _write_jsonl(candidate_root / "per_example.jsonl", per_example)
            result = {
                "evidence_schema_version": "1.0",
                "measured_at_utc": _utc_now(),
                "started_at_utc": started_at,
                "status": "completed",
                "study_id": prereg["study_id"],
                "dataset": label,
                "partition": "dev_pilot",
                "pilot_rows": len(rows),
                "candidate": candidate_id,
                "candidate_hash": logical_hash,
                "declared_delta": candidate["delta"],
                "config_path": _relative(config_path),
                "config_sha256": sha256_file(str(config_path)),
                "predictions_path": _relative(predictions_path),
                "predictions_sha256": sha256_file(str(predictions_path)),
                "selected_indices_sha256": _selected_indices_digest(predictions),
                "implementation_commit": _git_commit(),
                "dependency_versions": _dependency_versions(),
                "platform": platform.platform(),
                "execution": {
                    "workers": workers,
                    "worker_kind": "bounded_row_processes",
                    "omp_threads_per_worker": 1,
                    "embedding_cache_root": _relative(cache_root),
                    "elapsed_seconds": elapsed,
                },
                "dataset_preflight": dataset_preflight,
                "metrics": metrics,
                "dev_test_accessed": False,
                "test_split_accessed": False,
            }
        except Exception as error:
            result = {
                "evidence_schema_version": "1.0",
                "measured_at_utc": _utc_now(),
                "started_at_utc": started_at,
                "status": "failed",
                "study_id": prereg["study_id"],
                "dataset": label,
                "partition": "dev_pilot",
                "candidate": candidate_id,
                "candidate_hash": logical_hash,
                "declared_delta": candidate["delta"],
                "failure": f"{type(error).__name__}: {error}",
                "dev_test_accessed": False,
                "test_split_accessed": False,
            }
        _write_json(summary_path, result)
        _append_search_log({
            "logged_at_utc": _utc_now(),
            "study_id": prereg["study_id"],
            "dataset": label,
            "partition": "dev_pilot",
            "family": candidate["family"],
            "candidate": candidate_id,
            "candidate_hash": logical_hash,
            "config_path": _relative(config_path),
            "config_hash": sha256_file(str(config_path)),
            "dev_score": result.get("metrics", {}).get("macro_rouge"),
            "dev_test_score": None,
            "status": result["status"],
            "promoted": None,
            "reason": "D3a pilot screen; full-dev finalist rule not yet applied",
            "comparison_family_size": 83,
            "test_split_accessed": False,
        })
        results[candidate_id] = result
        if result["status"] != "completed":
            break

    status = (
        "completed"
        if len(results) == 14 and all(row["status"] == "completed" for row in results.values())
        else "partial_or_failed"
    )
    summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": status,
        "study_id": prereg["study_id"],
        "dataset": label,
        "partition": "dev_pilot",
        "pilot_rows": len(rows),
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": prereg_sha,
        "candidates": results,
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
    summary = run(args.dataset, workers=args.workers, resume=args.resume)
    print(json.dumps({"status": summary["status"], "dataset": summary["dataset"]}))


if __name__ == "__main__":
    main()
