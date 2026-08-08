"""Run the preregistered F-51 cached-vs-uncached equivalence audit on dev.

There is intentionally no dataset or split argument.  This script can only
read the frozen Multi-News dev partition declared by the preregistration; it
has no dev-test or test code path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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
from src.models.extractive.encoder_rank import (
    EMBEDDING_CACHE_CONTRACT_VERSION,
    EMBEDDING_CACHE_ENV,
)
from src.utils.io import read_jsonl


PREREGISTRATION = "configs/preregistrations/f51_embedding_cache_equivalence_v1.json"
PREREGISTRATION_SHA256 = "ccdd5a66c647c7a0448897ba8f4759fef356dee84d0b24eabf04d606bec66d98"


def _load_protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    actual = sha256_file(str(path))
    if actual != PREREGISTRATION_SHA256:
        raise ValueError(f"F-51 preregistration drifted: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_cached_rerun":
        raise ValueError("F-51 protocol is not frozen before the cached rerun")
    if protocol.get("partition") != "dev":
        raise ValueError("F-51 audit may read only frozen dev")
    if protocol.get("dev_test_accessed") is not False:
        raise ValueError("F-51 protocol must prohibit dev-test")
    if protocol.get("test_split_prohibited") is not True:
        raise ValueError("F-51 protocol must prohibit test")
    return protocol


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in read_jsonl(str(path))]


def _row_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("equivalence row has no string ID")
        if row_id in mapped:
            raise ValueError(f"equivalence artifact repeats ID {row_id!r}")
        mapped[row_id] = row
    return mapped


def _diagnostic_hashes(row: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = row.get("baseline_diagnostics") or {}
    return {
        "eligible_original_indices_sha256": diagnostics.get(
            "eligible_original_indices_sha256"
        ),
        "centroid_relevance_sha256": diagnostics.get(
            "centroid_relevance_sha256"
        ),
        "similarity_sha256": diagnostics.get("similarity_sha256"),
    }


def _compare(
    reference_rows: list[dict[str, Any]],
    cached_rows: list[dict[str, Any]],
    ordered_ids: list[str],
) -> dict[str, Any]:
    reference = _row_map(reference_rows)
    cached = _row_map(cached_rows)
    if set(reference) != set(ordered_ids) or set(cached) != set(ordered_ids):
        raise ValueError("reference/cached IDs do not equal the frozen dev partition")
    mismatch_counts = {
        "selected_indices": 0,
        "summary": 0,
        "feasibility": 0,
        "representation_hashes": 0,
    }
    mismatch_examples: dict[str, list[str]] = {
        key: [] for key in mismatch_counts
    }
    for row_id in ordered_ids:
        left = reference[row_id]
        right = cached[row_id]
        comparisons = {
            "selected_indices": left.get("selected_indices")
            == right.get("selected_indices"),
            "summary": left.get("summary") == right.get("summary"),
            "feasibility": (
                left.get("feasible"),
                left.get("infeasible_code"),
            )
            == (
                right.get("feasible"),
                right.get("infeasible_code"),
            ),
            "representation_hashes": _diagnostic_hashes(left)
            == _diagnostic_hashes(right),
        }
        for field, equal in comparisons.items():
            if not equal:
                mismatch_counts[field] += 1
                if len(mismatch_examples[field]) < 20:
                    mismatch_examples[field].append(row_id)
    return {
        "rows": len(ordered_ids),
        "mismatch_counts": mismatch_counts,
        "mismatch_examples_first_20": mismatch_examples,
        "all_row_contracts_exact": all(value == 0 for value in mismatch_counts.values()),
    }


def _append_registry(result: Mapping[str, Any]) -> None:
    attempt = "cached_full_dev"
    if any(
        row.get("study_id") == "f51-embedding-cache-equivalence-v1"
        and row.get("run_attempt") == attempt
        for row in _load_search_log()
    ):
        return
    passed = result.get("status") == "passed"
    _append_search_log(
        {
            "logged_at_utc": result["measured_at_utc"],
            "study_id": "f51-embedding-cache-equivalence-v1",
            "dataset": "Multi-News",
            "partition": "dev",
            "family": "engineering_equivalence",
            "candidate": "sbert_centroid_cached_vs_uncached",
            "method": "sbert_centroid",
            "candidate_hash": result["audit_hash"],
            "config_path": result["config_path"],
            "config_hash": result["config_sha256"],
            "run_attempt": attempt,
            "dev_score": result.get("cached_metrics", {}).get("macro_rouge"),
            "dev_test_score": None,
            "status": "completed" if passed else "failed",
            "promoted": passed,
            "reason": (
                "exact full-dev cache equivalence passed"
                if passed
                else "exact full-dev cache equivalence failed"
            ),
            "comparison_family_size": 0,
            "dev_test_accessed": False,
            "test_split_accessed": False,
        }
    )


def run() -> dict[str, Any]:
    protocol = _load_protocol()
    input_path = REPO_ROOT / protocol["input_path"]
    manifest_path = REPO_ROOT / protocol["partition_manifest_path"]
    config_path = REPO_ROOT / protocol["reference"]["config_path"]
    reference_predictions = REPO_ROOT / protocol["reference"]["predictions_path"]
    reference_evidence_path = REPO_ROOT / protocol["reference"]["evidence_path"]
    output_root = REPO_ROOT / protocol["cached_rerun"]["output_root"]
    cache_root = REPO_ROOT / protocol["cached_rerun"]["cache_root"]

    for path, expected in (
        (input_path, protocol["input_sha256"]),
        (manifest_path, protocol["partition_manifest_sha256"]),
        (config_path, protocol["reference"]["config_sha256"]),
        (reference_predictions, protocol["reference"]["predictions_sha256"]),
    ):
        if sha256_file(str(path)) != expected:
            raise ValueError(f"frozen F-51 input drifted: {_relative(path)}")
    if output_root.exists():
        raise ValueError(f"refusing to overwrite F-51 audit: {output_root}")
    if protocol["cached_rerun"]["cache_must_be_empty_before_run"]:
        if cache_root.exists() and any(cache_root.rglob("*")):
            raise ValueError(f"F-51 audit requires an empty cache root: {cache_root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ordered_ids = list(manifest["partitions"]["dev"]["selected_ids"])
    if len(ordered_ids) != protocol["partition_rows"]:
        raise ValueError("F-51 frozen partition row count drifted")
    if selected_ids_sha256(ordered_ids) != protocol["partition_selected_ids_sha256"]:
        raise ValueError("F-51 frozen selected-ID digest drifted")
    reference_evidence = json.loads(reference_evidence_path.read_text(encoding="utf-8"))
    if (
        reference_evidence.get("selected_indices_sha256")
        != protocol["reference"]["selected_indices_sha256"]
    ):
        raise ValueError("F-51 reference selected-index evidence drifted")

    gold = _load_gold(input_path, ordered_ids)
    audit_hash = _canonical_sha256(
        {
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "reference_config_sha256": protocol["reference"]["config_sha256"],
            "cache_contract_version": EMBEDDING_CACHE_CONTRACT_VERSION,
        }
    )
    context = {
        "study_id": protocol["study_id"],
        "dataset": protocol["dataset"],
        "partition": "dev",
        "partition_rows": len(ordered_ids),
        "partition_manifest_path": protocol["partition_manifest_path"],
        "partition_manifest_sha256": protocol["partition_manifest_sha256"],
        "partition_selected_ids_sha256": protocol["partition_selected_ids_sha256"],
        "preregistration_path": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "audit_hash": audit_hash,
        "dev_test_accessed": False,
        "execution_optimization": {
            "embedding_cache": {
                "enabled": True,
                "root": _relative(cache_root),
                "contract_version": EMBEDDING_CACHE_CONTRACT_VERSION,
                "scientific_config_unchanged": True,
            }
        },
    }

    cold_root = output_root / protocol["cached_rerun"]["cold_populate_run"]
    warm_root = output_root / protocol["cached_rerun"]["warm_hit_run"]
    try:
        cold_metrics, _ = _run_method(
            "sbert_centroid",
            config_path=config_path,
            config_sha256=protocol["reference"]["config_sha256"],
            input_path=input_path,
            candidate_root=cold_root,
            ordered_ids=ordered_ids,
            gold=gold,
            study_context=context,
            subprocess_env={EMBEDDING_CACHE_ENV: str(cache_root)},
        )
        warm_metrics, _ = _run_method(
            "sbert_centroid",
            config_path=config_path,
            config_sha256=protocol["reference"]["config_sha256"],
            input_path=input_path,
            candidate_root=warm_root,
            ordered_ids=ordered_ids,
            gold=gold,
            study_context=context,
            subprocess_env={EMBEDDING_CACHE_ENV: str(cache_root)},
        )
    except Exception as error:
        failure = {
            "evidence_schema_version": "1.0",
            "measured_at_utc": _utc_now(),
            "status": "failed",
            "study_id": protocol["study_id"],
            "implementation_commit": _git_commit(),
            "audit_hash": audit_hash,
            "config_path": protocol["reference"]["config_path"],
            "config_sha256": protocol["reference"]["config_sha256"],
            "failure": f"{type(error).__name__}: {error}",
            "dev_test_accessed": False,
            "test_split_accessed": False,
        }
        _write_json(output_root / "equivalence_summary.json", failure)
        _append_registry(failure)
        raise
    cold_run = cold_root / "sbert_centroid" / "run"
    warm_run = warm_root / "sbert_centroid" / "run"
    cold_predictions = cold_run / "predictions.jsonl"
    warm_predictions = warm_run / "predictions.jsonl"
    cold_evidence_path = cold_run / "evidence.json"
    warm_evidence_path = warm_run / "evidence.json"
    cold_evidence = json.loads(cold_evidence_path.read_text(encoding="utf-8"))
    warm_evidence = json.loads(warm_evidence_path.read_text(encoding="utf-8"))
    reference_rows = _load_rows(reference_predictions)
    cold_comparison = _compare(
        reference_rows,
        _load_rows(cold_predictions),
        ordered_ids,
    )
    warm_comparison = _compare(
        reference_rows,
        _load_rows(warm_predictions),
        ordered_ids,
    )
    cold_metrics_exact = cold_metrics == reference_evidence["metrics"]
    warm_metrics_exact = warm_metrics == reference_evidence["metrics"]
    cold_cache_summary = cold_evidence.get("embedding_cache_summary") or {}
    warm_cache_summary = warm_evidence.get("embedding_cache_summary") or {}
    cold_cache_provenance_exact = (
        cold_cache_summary.get("rows") == len(ordered_ids)
        and sum(cold_cache_summary.get("status_counts", {}).values())
        == len(ordered_ids)
        and cold_cache_summary.get("contract_version")
        == protocol["cached_rerun"]["cache_contract_version"]
    )
    warm_cache_provenance_exact = (
        warm_cache_summary.get("rows") == len(ordered_ids)
        and warm_cache_summary.get("status_counts") == {"hit": len(ordered_ids)}
        and warm_cache_summary.get("contract_version")
        == protocol["cached_rerun"]["cache_contract_version"]
    )
    selected_digest_exact = all(
        evidence.get("selected_indices_sha256")
        == protocol["reference"]["selected_indices_sha256"]
        for evidence in (cold_evidence, warm_evidence)
    )
    passed = bool(
        cold_comparison["all_row_contracts_exact"]
        and warm_comparison["all_row_contracts_exact"]
        and cold_metrics_exact
        and warm_metrics_exact
        and cold_cache_provenance_exact
        and warm_cache_provenance_exact
        and selected_digest_exact
    )
    result = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "passed" if passed else "failed",
        "study_id": protocol["study_id"],
        "implementation_commit": _git_commit(),
        "audit_hash": audit_hash,
        "config_path": protocol["reference"]["config_path"],
        "config_sha256": protocol["reference"]["config_sha256"],
        "reference_evidence_path": protocol["reference"]["evidence_path"],
        "reference_evidence_sha256": sha256_file(str(reference_evidence_path)),
        "cold_evidence_path": _relative(cold_evidence_path),
        "cold_evidence_sha256": sha256_file(str(cold_evidence_path)),
        "warm_evidence_path": _relative(warm_evidence_path),
        "warm_evidence_sha256": sha256_file(str(warm_evidence_path)),
        "reference_selected_indices_sha256": protocol["reference"][
            "selected_indices_sha256"
        ],
        "cold_selected_indices_sha256": cold_evidence.get("selected_indices_sha256"),
        "warm_selected_indices_sha256": warm_evidence.get("selected_indices_sha256"),
        "cold_row_comparison": cold_comparison,
        "warm_row_comparison": warm_comparison,
        "cold_metrics_exact": cold_metrics_exact,
        "warm_metrics_exact": warm_metrics_exact,
        "reference_metrics": reference_evidence["metrics"],
        "cold_metrics": cold_metrics,
        "cached_metrics": warm_metrics,
        "cold_cache_provenance_exact": cold_cache_provenance_exact,
        "warm_cache_provenance_exact": warm_cache_provenance_exact,
        "cold_embedding_cache_summary": cold_cache_summary,
        "warm_embedding_cache_summary": warm_cache_summary,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(output_root / "equivalence_summary.json", result)
    _append_registry(result)
    if not passed:
        raise RuntimeError("F-51 exact embedding-cache equivalence audit failed")
    return result


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
