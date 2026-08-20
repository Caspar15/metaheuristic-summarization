"""Score-preserving recovery after the frozen final runner's aggregation bug.

This script may only read already-completed prediction/evaluator checkpoints. It
does not run a selector, tokenize text, invoke ROUGE, or alter any source score.
The sole repair derives each internal per-row macro as the arithmetic mean of
its already-frozen R1/R2/R-Lsum values, then completes the preregistered analysis.
"""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from statistics import fmean
from typing import Any, Mapping, Sequence

from scripts.audit.run_govreport_final_test import (
    DEFAULT_ACTIVATION,
    DEFAULT_FREEZE,
    DETERMINISTIC_BASELINES,
    FAMILY_LABELS,
    OUTPUT_ROOT,
    RANDOM_SEEDS,
    REPO_ROOT,
    _analyze_official,
    _dataset_identity,
    _git_commit,
    _load_activation,
    _load_freeze,
    _relative,
    _utc_now,
    _write_json,
    _write_jsonl,
)
from src.data.policy import sha256_file
from src.eval.rouge import DEFAULT_METRICS
from src.utils.io import read_jsonl


FREEZE_SHA256 = "5dbd5490d9d3616314cf32bf69efcbe0dfdb38dfb91a4d082a67c1a9166d7d92"
ACTIVATION_SHA256 = "5b9dd5ba54ab5c67679d66d8e1e7c3b47e184497bcc6a2da0bd8bc6c51d2a8b6"
FAILED_ATTEMPT = Path(
    "runs_v2/govreport_final_test_v1/attempts/"
    "attempt_20260815T234414Z_failed/evidence.json"
)


def derive_internal_random_rows(
    seed_rows: Sequence[Mapping[str, Mapping[str, float]]],
    ordered_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Aggregate fixed seeds and derive macro from the three source metrics."""

    if len(seed_rows) != len(RANDOM_SEEDS):
        raise ValueError(f"expected {len(RANDOM_SEEDS)} random seeds")
    output: list[dict[str, Any]] = []
    for row_id in ordered_ids:
        metrics = {
            metric: fmean(float(seed[row_id][metric]) for seed in seed_rows)
            for metric in DEFAULT_METRICS
        }
        output.append(
            {"id": row_id, **metrics, "macro_rouge": fmean(metrics.values())}
        )
    return output


def _load_completed(protocol: str, label: str, ordered_ids: Sequence[str]) -> dict[str, Any]:
    evidence_path = REPO_ROOT / OUTPUT_ROOT / protocol / label / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "completed" or evidence.get("rows") != 973:
        raise ValueError(f"incomplete frozen checkpoint: {protocol}/{label}")
    per_path = REPO_ROOT / evidence["per_example_path"]
    if sha256_file(str(per_path)) != evidence["per_example_sha256"]:
        raise ValueError(f"per-example hash drifted: {protocol}/{label}")
    rows = list(read_jsonl(str(per_path)))
    if [row["id"] for row in rows] != list(ordered_ids):
        raise ValueError(f"row order drifted: {protocol}/{label}")
    return evidence


def _recover_internal_random(
    seed_results: Mapping[str, Mapping[str, Any]], ordered_ids: Sequence[str]
) -> dict[str, Any]:
    seed_rows: list[dict[str, Mapping[str, float]]] = []
    for label in (f"random_seed_{seed}" for seed in RANDOM_SEEDS):
        evidence = seed_results[label]
        rows = list(read_jsonl(str(REPO_ROOT / evidence["per_example_path"])))
        seed_rows.append({str(row["id"]): row for row in rows})
    aggregated = derive_internal_random_rows(seed_rows, ordered_ids)
    root = REPO_ROOT / OUTPUT_ROOT / "internal/random_mean_10_seeds"
    per_path = root / "per_example.jsonl"
    _write_jsonl(per_path, aggregated)
    corpus = {
        metric: fmean(float(result["metrics"][metric]) for result in seed_results.values())
        for metric in (*DEFAULT_METRICS, "macro_rouge")
    }
    evidence = {
        "status": "completed",
        "system": "random_mean_10_seeds",
        "protocol": "internal_multisentence_lsum_secondary",
        "rows": len(ordered_ids),
        "seeds": list(RANDOM_SEEDS),
        "metrics": corpus,
        "per_example_path": _relative(per_path),
        "per_example_sha256": sha256_file(str(per_path)),
        "recovery": "per-row macro derived as mean(rouge1, rouge2, rougeLsum)",
        "source_scores_modified": False,
        "test_split_accessed": True,
        "test_scores_observed": True,
    }
    _write_json(root / "evidence.json", evidence)
    return evidence


def main() -> None:
    freeze = _load_freeze(DEFAULT_FREEZE, FREEZE_SHA256)
    _load_activation(DEFAULT_ACTIVATION, ACTIVATION_SHA256, FREEZE_SHA256)
    _, ordered_ids, _ = _dataset_identity(freeze)
    failed = json.loads((REPO_ROOT / FAILED_ATTEMPT).read_text(encoding="utf-8"))
    if failed.get("failure") != "KeyError: 'macro_rouge'":
        raise ValueError("frozen failure identity drifted")

    labels = [
        "proposed",
        *DETERMINISTIC_BASELINES,
        *(f"random_seed_{seed}" for seed in RANDOM_SEEDS),
    ]
    official = {label: _load_completed("official", label, ordered_ids) for label in labels}
    internal = {label: _load_completed("internal", label, ordered_ids) for label in labels}
    official_random = _load_completed("official", "random_mean_10_seeds", ordered_ids)
    internal_random = _recover_internal_random(
        {label: value for label, value in internal.items() if label.startswith("random_seed_")},
        ordered_ids,
    )
    official["random_mean_10_seeds"] = official_random
    internal["random_mean_10_seeds"] = internal_random
    family_official = {label: official[label] for label in FAMILY_LABELS}
    family_internal = {label: internal[label] for label in FAMILY_LABELS}

    output = REPO_ROOT / OUTPUT_ROOT
    analysis = _analyze_official(family_official, ordered_ids)
    analysis["completion_mode"] = "score_preserving_post_failure_recovery"
    analysis["source_scores_modified"] = False
    _write_json(output / "analysis.json", analysis)
    _write_json(
        output / "internal_summary.json",
        {
            "status": "completed",
            "protocol": "internal_multisentence_lsum_secondary",
            "completion_mode": "score_preserving_post_failure_recovery",
            "corpus_ranking": sorted(
                ({"system": label, **dict(value["metrics"])} for label, value in family_internal.items()),
                key=lambda row: (-float(row["macro_rouge"]), row["system"]),
            ),
            "source_scores_modified": False,
            "test_scores_observed": True,
        },
    )
    execution = {
        "status": "completed_via_score_preserving_recovery",
        "study_id": "govreport-centered-final-evaluation-v1",
        "completed_at_utc": _utc_now(),
        "freeze_manifest_sha256": FREEZE_SHA256,
        "activation_sha256": ACTIVATION_SHA256,
        "scientific_code_commit": freeze["scientific_code_commit"],
        "recovery_commit": _git_commit(),
        "failed_attempt_path": FAILED_ATTEMPT.as_posix(),
        "failed_attempt_sha256": sha256_file(str(REPO_ROOT / FAILED_ATTEMPT)),
        "recovery_scope": "derive missing internal random per-row macro and complete frozen analyses only",
        "predictions_rerun": False,
        "official_evaluator_rerun": False,
        "source_scores_modified": False,
        "platform": platform.platform(),
        "python": sys.version,
        "dependencies": {name: version(name) for name in ("numpy", "rouge-score")},
        "analysis_path": _relative(output / "analysis.json"),
        "analysis_sha256": sha256_file(str(output / "analysis.json")),
        "internal_summary_path": _relative(output / "internal_summary.json"),
        "internal_summary_sha256": sha256_file(str(output / "internal_summary.json")),
        "decision": analysis["decision"],
        "post_score_tuning_permitted": False,
        "test_split_accessed": True,
        "test_scores_observed": True,
    }
    _write_json(output / "execution_evidence.json", execution)
    print(json.dumps({"status": execution["status"], "decision": analysis["decision"]}))


if __name__ == "__main__":
    main()
