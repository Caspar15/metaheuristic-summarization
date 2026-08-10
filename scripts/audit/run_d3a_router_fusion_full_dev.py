"""Run mechanically selected D3a finalists on full frozen dev only.

There is no split argument.  Candidate rows are streamed through at most 16
bounded workers and full prediction artifacts are written atomically, avoiding
the RAM spike caused by retaining all candidate provenance in memory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import yaml

from scripts.audit.run_d3a_router_fusion_screen import _bounded_rows
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
    _write_jsonl,
)
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file, validate_dataset_policy_request
from src.eval.rouge import DEFAULT_METRICS, rouge_scores
from src.pipeline.select_sentences import validate_requested_split
from src.utils.io import load_yaml, read_jsonl


PREREGISTRATION = Path("configs/preregistrations/d3a_router_fusion_full_dev_v1.json")
PREREGISTRATION_SHA256 = (
    "083b3a615dc10169bf14557878853abf0c8ebc82ca345308df9e57d58f7fc5c5"
)
DATASET_LABELS = {"multinews": "Multi-News", "govreport": "GovReport"}


def _protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    if sha256_file(str(path)) != PREREGISTRATION_SHA256:
        raise ValueError("D3a full-dev preregistration SHA-256 drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("partition") != "dev" or value.get("dev_test_access") != "none":
        raise ValueError("D3a confirmation must remain on frozen dev")
    if value.get("test_split_prohibited") is not True:
        raise ValueError("D3a confirmation must prohibit test")
    pilot_path = REPO_ROOT / value["source_pilot_analysis"]
    if sha256_file(str(pilot_path)) != value["source_pilot_analysis_sha256"]:
        raise ValueError("D3a pilot finalist analysis drifted")
    return value


def _set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node: dict[str, Any] = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"D3a full-dev delta parent {part!r} is absent")
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
        raise ValueError("D3a full-dev route contract drifted")
    if config["selector"].get("salience_source") != "rrf_fusion":
        raise ValueError("D3a full-dev salience contract drifted")
    if config["selector"].get("similarity_source") != "pipeline_similarity":
        raise ValueError("D3a full-dev selector representation drifted")
    return config


def _selected_rows(input_path: Path, ordered_ids: Sequence[str]) -> Iterator[dict[str, Any]]:
    wanted = set(ordered_ids)
    expected_position = 0
    for row in read_jsonl(str(input_path)):
        row_id = row.get("id")
        if row_id not in wanted:
            continue
        if expected_position >= len(ordered_ids) or row_id != ordered_ids[expected_position]:
            raise ValueError("D3a frozen dev IDs are not in canonical input order")
        validate_requested_split(row, "validation")
        expected_position += 1
        yield row
    if expected_position != len(ordered_ids):
        raise ValueError("D3a full-dev input is missing frozen IDs")


def _selected_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            {"id": row["id"], "selected_indices": row["selected_indices"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _anchor_result(registration: Mapping[str, Any], expected_rows: int) -> dict[str, Any]:
    run = REPO_ROOT / registration["anchor_run"]
    metrics_path = run / "metrics.json"
    evidence_path = run / "evidence.json"
    per_example_path = run / "per_example.jsonl"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if int(metrics.get("rows", -1)) != expected_rows:
        raise ValueError("D3a reused anchor row count drifted")
    if evidence.get("dev_test_accessed") is not False or evidence.get("test_split_accessed") is not False:
        raise ValueError("D3a reused anchor lacks protected-split guards")
    return {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed_reused_anchor",
        "candidate": "R00_anchor",
        "metrics": metrics,
        "run_path": _relative(run),
        "per_example_path": _relative(per_example_path),
        "per_example_sha256": sha256_file(str(per_example_path)),
        "evidence_path": _relative(evidence_path),
        "evidence_sha256": sha256_file(str(evidence_path)),
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }


def _run_candidate(
    *,
    rows: Iterator[dict[str, Any]],
    ordered_ids: Sequence[str],
    gold: Mapping[str, Sequence[str]],
    config: dict[str, Any],
    candidate_root: Path,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions_path = candidate_root / "predictions.jsonl"
    temporary = predictions_path.with_suffix(".jsonl.tmp")
    summaries: list[str] = []
    feasibility: list[bool] = []
    lengths: list[int] = []
    selected_rows: list[dict[str, Any]] = []
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for position, prediction in enumerate(_bounded_rows(rows, config, workers=workers)):
            if prediction.get("id") != ordered_ids[position]:
                raise RuntimeError("D3a full-dev prediction order drifted")
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            summaries.append(str(prediction.get("summary", "")))
            feasibility.append(bool(prediction.get("feasible")))
            lengths.append(len(str(prediction.get("summary", "")).split()))
            selected_rows.append({
                "id": prediction["id"],
                "selected_indices": prediction["selected_indices"],
            })
        handle.flush()
        os.fsync(handle.fileno())
    if len(summaries) != len(ordered_ids):
        raise RuntimeError("D3a full-dev prediction count drifted")
    os.replace(temporary, predictions_path)
    references = [gold[row_id] for row_id in ordered_ids]
    means, per_example = rouge_scores(
        summaries, references, metrics=DEFAULT_METRICS, return_per_example=True
    )
    per_example_rows = [
        {"id": row_id, **scores}
        for row_id, scores in zip(ordered_ids, per_example)
    ]
    _write_jsonl(candidate_root / "per_example.jsonl", per_example_rows)
    metrics = {
        "evaluation_protocol": "multisentence_lsum",
        "rows": len(ordered_ids),
        "rouge": means,
        "macro_rouge": fmean(float(means[name]) for name in DEFAULT_METRICS),
        "feasible_rows": int(sum(feasibility)),
        "infeasible_rows": int(len(feasibility) - sum(feasibility)),
        "summary_words": {
            "mean": float(np.mean(lengths)),
            "min": int(min(lengths)),
            "max": int(max(lengths)),
        },
    }
    _write_json(candidate_root / "metrics.json", metrics)
    artifacts = {
        "predictions_path": _relative(predictions_path),
        "predictions_sha256": sha256_file(str(predictions_path)),
        "selected_indices_sha256": _selected_digest(selected_rows),
        "per_example_path": _relative(candidate_root / "per_example.jsonl"),
        "per_example_sha256": sha256_file(str(candidate_root / "per_example.jsonl")),
    }
    return metrics, artifacts


def run(dataset: str, *, workers: int, resume: bool = False) -> dict[str, Any]:
    if dataset not in DATASET_LABELS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    label = DATASET_LABELS[dataset]
    protocol = _protocol()
    registration = protocol["datasets"][label]
    spec = _load_study(dataset)
    input_path = REPO_ROOT / spec["input"]
    base_path = REPO_ROOT / registration["base_config"]
    if sha256_file(str(base_path)) != registration["base_config_sha256"]:
        raise ValueError("D3a full-dev base config drifted")
    base = load_yaml(str(base_path))
    validate_dataset_policy_request(base, str(input_path), "validation")
    partition = spec["manifest_object"]["partitions"]["dev"]
    ordered_ids = [str(value) for value in partition["selected_ids"]]
    if len(ordered_ids) != int(registration["rows"]):
        raise ValueError("D3a full-dev row count drifted")
    if selected_ids_sha256(ordered_ids) != partition["selected_ids_sha256"]:
        raise ValueError("D3a full-dev ID digest drifted")
    gold = _load_gold(input_path, ordered_ids)
    output_root = REPO_ROOT / "runs_v2/d3a_router_fusion_full_dev_v1" / dataset / "dev"
    if output_root.exists() and not resume:
        raise ValueError(f"refusing to overwrite D3a full-dev study: {output_root}")
    output_root.mkdir(parents=True, exist_ok=resume)
    cache_root = REPO_ROOT / "runs_v2/gate2_baseline_matrix_v1" / dataset / "dev/plm/_embedding_cache"
    if not cache_root.is_dir():
        raise ValueError("D3a full-dev embedding cache is missing")
    os.environ["META_SUM_EMBEDDING_CACHE_DIR"] = str(cache_root)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    results: dict[str, Any] = {}
    for candidate_id in registration["finalists"]:
        candidate = protocol["candidates"][candidate_id]
        if candidate.get("reuse_anchor"):
            results[candidate_id] = _anchor_result(registration, len(ordered_ids))
            continue
        candidate_root = output_root / candidate_id
        summary_path = candidate_root / "candidate_summary.json"
        logical_hash = _canonical_sha256({
            "study_id": protocol["study_id"],
            "dataset": label,
            "candidate_id": candidate_id,
            "candidate": candidate,
            "base_config_sha256": registration["base_config_sha256"],
            "selected_ids_sha256": partition["selected_ids_sha256"],
        })
        if summary_path.is_file():
            if not resume:
                raise ValueError(f"D3a full-dev candidate exists: {candidate_root}")
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
            if prior.get("candidate_hash") != logical_hash:
                raise ValueError("D3a full-dev resume hash drifted")
            if prior.get("status") == "completed":
                results[candidate_id] = prior
                continue
            raise ValueError("failed D3a attempt is preserved; archive before retry")
        candidate_root.mkdir(parents=True, exist_ok=False)
        config = resolve_candidate(base, candidate)
        config["study"] = {
            "study_id": protocol["study_id"],
            "family": candidate["family"],
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
        started_at = _utc_now()
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
                "declared_delta": candidate["delta"],
                "config_path": _relative(config_path),
                "config_sha256": sha256_file(str(config_path)),
                "implementation_commit": _git_commit(),
                "dependency_versions": _dependency_versions(),
                "execution": {
                    "workers": workers,
                    "worker_kind": "bounded_row_processes",
                    "threads_per_worker": 1,
                    "embedding_cache_root": _relative(cache_root),
                    "streamed_predictions": True,
                },
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
            "family": candidate["family"],
            "candidate": candidate_id,
            "candidate_hash": logical_hash,
            "config_path": _relative(config_path),
            "config_hash": sha256_file(str(config_path)),
            "dev_score": result.get("metrics", {}).get("macro_rouge"),
            "dev_test_score": None,
            "status": result["status"],
            "promoted": None,
            "reason": "D3a full-dev confirmation; paired adversarial gate pending",
            "comparison_family_size": 83,
            "test_split_accessed": False,
        })
        results[candidate_id] = result
        if result["status"] != "completed":
            break
    expected = len(registration["finalists"])
    status = (
        "completed"
        if len(results) == expected and all(str(row["status"]).startswith("completed") for row in results.values())
        else "partial_or_failed"
    )
    summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": status,
        "study_id": protocol["study_id"],
        "dataset": label,
        "partition": "dev",
        "partition_rows": len(ordered_ids),
        "finalists": list(registration["finalists"]),
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
