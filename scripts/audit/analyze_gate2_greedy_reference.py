"""Analyze preregistered greedy-reference headroom and candidate recall on dev.

The CLI has no split option. Inputs, formulas, empty-set handling, and labels
are pinned by gate2_greedy_reference_analysis_v1 before overlap scores existed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

from scripts.audit.run_gate2_greedy_reference import REPO_ROOT
from scripts.audit.run_length_contract_study import (
    _append_search_log,
    _relative,
    _utc_now,
    _write_json,
)
from src.data.policy import sha256_file
from src.eval.rouge import DEFAULT_METRICS
from src.utils.io import read_jsonl


PREREGISTRATION = "configs/preregistrations/gate2_greedy_reference_analysis_v2.json"
PREREGISTRATION_SHA256 = "ef45c0562a29569709d2e7700da0f5570fe8c39b06ddb9b87166ca7fc66a3e39"
BASE_PREREGISTRATION = "configs/preregistrations/gate2_greedy_reference_analysis_v1.json"
BASE_PREREGISTRATION_SHA256 = "a33b0ec91f944ac2d8836cc8c46f24898fe9041b66a3da1bbd4c556e9c774669"
ROUTES = ("lexical", "semantic", "graph")


def _load_protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    actual = sha256_file(str(path))
    if actual != PREREGISTRATION_SHA256:
        raise ValueError(f"greedy-reference analysis preregistration drifted: {actual}")
    overlay = json.loads(path.read_text(encoding="utf-8"))
    base_path = REPO_ROOT / BASE_PREREGISTRATION
    if sha256_file(str(base_path)) != BASE_PREREGISTRATION_SHA256:
        raise ValueError("base greedy-reference analysis preregistration drifted")
    if overlay.get("supersedes", {}).get("sha256") != BASE_PREREGISTRATION_SHA256:
        raise ValueError("analysis-v2 does not pin the superseded v1 protocol")
    if overlay["supersedes"].get("overlap_scores_accessed_before_correction") is not False:
        raise ValueError("route-pool correction was not frozen before overlap scores")
    protocol = json.loads(base_path.read_text(encoding="utf-8"))
    protocol.update(
        {
            "study_id": overlay["study_id"],
            "status": overlay["status"],
            "candidate_recall": overlay["candidate_recall"],
        }
    )
    if overlay.get("status") != "frozen_before_candidate_overlap_scores":
        raise ValueError("candidate-overlap analysis was not frozen before scores")
    if protocol.get("partition") != "dev":
        raise ValueError("candidate-overlap analysis must use dev")
    if protocol.get("dev_test_accessed") is not False:
        raise ValueError("candidate-overlap analysis must not read dev-test")
    if protocol.get("test_split_prohibited") is not True:
        raise ValueError("candidate-overlap analysis must prohibit test")
    return protocol


def _checked_jsonl(path_spec: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    path = REPO_ROOT / str(path_spec[f"{key}_path"])
    expected = str(path_spec[f"{key}_sha256"])
    if sha256_file(str(path)) != expected:
        raise ValueError(f"analysis input drifted: {_relative(path)}")
    return [dict(row) for row in read_jsonl(str(path))]


def _require_ids(rows: Iterable[Mapping[str, Any]], expected: list[str], label: str) -> None:
    actual = [str(row.get("id")) for row in rows]
    if actual != expected:
        raise ValueError(f"{label} IDs are not the exact frozen order")


def _mean_metrics(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate empty per-example metrics")
    return {
        metric: fmean(float(row[metric]) for row in rows) for metric in DEFAULT_METRICS
    }


def _headroom_summary(
    lead: Mapping[str, float],
    greedy: Mapping[str, float],
    systems: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"systems": {}}
    for metric in DEFAULT_METRICS:
        if float(greedy[metric]) <= float(lead[metric]):
            raise ValueError(f"greedy reference is not above Lead for {metric}")
    for label, scores in systems.items():
        per_metric: dict[str, Any] = {}
        for metric in DEFAULT_METRICS:
            denominator = float(greedy[metric]) - float(lead[metric])
            captured = (float(scores[metric]) - float(lead[metric])) / denominator
            per_metric[metric] = {
                "lead": float(lead[metric]),
                "system": float(scores[metric]),
                "greedy_reference": float(greedy[metric]),
                "available_headroom": denominator,
                "captured_fraction": captured,
                "remaining_gap_to_greedy_reference": float(greedy[metric])
                - float(scores[metric]),
            }
        output["systems"][label] = {
            "per_metric": per_metric,
            "mean_captured_fraction": fmean(
                row["captured_fraction"] for row in per_metric.values()
            ),
        }
    return output


def _recall_summary(
    greedy_rows: list[Mapping[str, Any]],
    proposed_rows: list[Mapping[str, Any]],
    *,
    route_top_k: int,
    total_cap: int,
) -> dict[str, Any]:
    if len(greedy_rows) != len(proposed_rows):
        raise ValueError("greedy/proposed row counts differ")
    labels = ("union_pool", *ROUTES, "selected_set")
    hits = {label: 0 for label in labels}
    macro = {label: [] for label in labels}
    complete = {label: 0 for label in labels}
    exclusive_route_hits = {route: 0 for route in ROUTES}
    denominator = 0
    empty_rows = 0

    for greedy_row, proposed_row in zip(greedy_rows, proposed_rows):
        if str(greedy_row.get("id")) != str(proposed_row.get("id")):
            raise ValueError("greedy/proposed IDs differ")
        greedy_set = {int(value) for value in greedy_row["selected_indices"]}
        records = list(proposed_row.get("candidate_records", []))
        union_pool = {int(record["original_index"]) for record in records}
        if len(union_pool) != len(records):
            raise ValueError("candidate_records contain duplicate original_index")
        pool = proposed_row.get("candidate_pool", {})
        if int(pool.get("total_cap")) != total_cap:
            raise ValueError("candidate-pool total cap drifted")
        if int(pool.get("route_top_k")) != route_top_k:
            raise ValueError("candidate-pool route top-K drifted")
        if int(pool.get("actual_size")) != len(records) or len(records) > total_cap:
            raise ValueError("candidate-pool actual size violates configured cap")
        route_proposals = pool.get("route_proposals", {})
        if set(route_proposals) != set(ROUTES):
            raise ValueError("candidate_pool route proposals are incomplete")
        if any(len(route_proposals[route]) > route_top_k for route in ROUTES):
            raise ValueError("route proposal count exceeds route top-K")
        route_pools = {
            route: {
                int(proposal["original_index"])
                for proposal in route_proposals[route]
            }
            for route in ROUTES
        }
        sets = {
            "union_pool": union_pool,
            **route_pools,
            "selected_set": {int(value) for value in proposed_row["selected_indices"]},
        }
        if not greedy_set:
            empty_rows += 1
            continue
        denominator += len(greedy_set)
        for label, candidate_set in sets.items():
            intersection = len(greedy_set & candidate_set)
            hits[label] += intersection
            macro[label].append(intersection / len(greedy_set))
            complete[label] += int(greedy_set <= candidate_set)
        for index in greedy_set:
            memberships = [route for route in ROUTES if index in route_pools[route]]
            if len(memberships) == 1:
                exclusive_route_hits[memberships[0]] += 1

    nonempty_rows = len(greedy_rows) - empty_rows
    if denominator <= 0 or nonempty_rows <= 0:
        raise ValueError("greedy-reference recall denominator is empty")
    return {
        "rows": len(greedy_rows),
        "nonempty_greedy_rows": nonempty_rows,
        "empty_greedy_rows": empty_rows,
        "greedy_selected_denominator": denominator,
        "sets": {
            label: {
                "micro_recall": hits[label] / denominator,
                "macro_recall": fmean(macro[label]),
                "complete_row_rate": complete[label] / nonempty_rows,
                "hit_count": hits[label],
            }
            for label in labels
        },
        "exclusive_route_hits": exclusive_route_hits,
    }


def analyze_dataset(dataset: str) -> dict[str, Any]:
    protocol = _load_protocol()
    spec = protocol["datasets"][dataset]
    proposed_rows = _checked_jsonl(spec["proposed"], "predictions")
    proposed_metrics_rows = _checked_jsonl(spec["proposed"], "per_example")
    baseline_rows = _checked_jsonl(spec["strongest_completed_baseline"], "per_example")
    if len(proposed_rows) != int(spec["rows"]):
        raise ValueError("proposed row count differs from preregistration")
    frozen_ids = [str(row["id"]) for row in proposed_rows]
    _require_ids(proposed_metrics_rows, frozen_ids, "proposed per-example")
    _require_ids(baseline_rows, frozen_ids, "strongest baseline per-example")

    lead_path = REPO_ROOT / str(spec["lead_metrics"]["path"])
    if sha256_file(str(lead_path)) != spec["lead_metrics"]["sha256"]:
        raise ValueError("Lead metrics drifted")
    lead_artifact = json.loads(lead_path.read_text(encoding="utf-8"))
    if int(lead_artifact["rows"]) != int(spec["rows"]):
        raise ValueError("Lead row count differs from preregistration")

    greedy_scores: dict[str, float] = {}
    recalls: dict[str, Any] = {}
    greedy_evidence: dict[str, Any] = {}
    greedy_root = REPO_ROOT / str(spec["greedy_reference_root"])
    for target in DEFAULT_METRICS:
        target_root = greedy_root / target
        evidence_path = target_root / "evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") != "completed":
            raise ValueError(f"greedy-reference {target} is not completed")
        if evidence.get("optimization_target") != target:
            raise ValueError(f"greedy-reference target drifted: {target}")
        if evidence.get("dev_test_accessed") is not False:
            raise ValueError("greedy-reference evidence accessed dev-test")
        if evidence.get("test_split_accessed") is not False:
            raise ValueError("greedy-reference evidence accessed test")
        rows_path = REPO_ROOT / str(evidence["rows_path"])
        if sha256_file(str(rows_path)) != evidence["rows_sha256"]:
            raise ValueError(f"greedy-reference rows drifted: {target}")
        greedy_rows = [dict(row) for row in read_jsonl(str(rows_path))]
        _require_ids(greedy_rows, frozen_ids, f"greedy-reference {target}")
        greedy_scores[target] = float(evidence["metrics"]["rouge"][target])
        recalls[target] = _recall_summary(
            greedy_rows,
            proposed_rows,
            route_top_k=int(spec["proposed"]["route_top_k"]),
            total_cap=int(spec["proposed"]["total_cap"]),
        )
        greedy_evidence[target] = {
            "path": _relative(evidence_path),
            "sha256": sha256_file(str(evidence_path)),
            "rows_sha256": evidence["rows_sha256"],
            "selected_indices_sha256": evidence["selected_indices_sha256"],
        }

    proposed_scores = _mean_metrics(proposed_metrics_rows)
    baseline_scores = _mean_metrics(baseline_rows)
    analysis = {
        "analysis_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "study_id": protocol["study_id"],
        "status": "completed",
        "dataset": spec["label"],
        "partition": "dev",
        "rows": int(spec["rows"]),
        "headroom": _headroom_summary(
            lead_artifact["rouge"],
            greedy_scores,
            {
                spec["proposed"]["label"]: proposed_scores,
                spec["strongest_completed_baseline"]["label"]: baseline_scores,
            },
        ),
        "candidate_recall": recalls,
        "score_sources": {
            "lead": lead_artifact["rouge"],
            "greedy_reference_metric_specific": greedy_scores,
            "proposed": proposed_scores,
            "strongest_completed_baseline": baseline_scores,
        },
        "greedy_reference_evidence": greedy_evidence,
        "preregistration_path": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    output_root = greedy_root / "analysis"
    output_path = output_root / "analysis.json"
    evidence_path = output_root / "evidence.json"
    if output_path.exists() or evidence_path.exists():
        raise ValueError(f"refusing to overwrite greedy-reference analysis: {output_root}")
    _write_json(output_path, analysis)
    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed",
        "study_id": protocol["study_id"],
        "dataset": spec["label"],
        "partition": "dev",
        "rows": int(spec["rows"]),
        "analysis_path": _relative(output_path),
        "analysis_sha256": sha256_file(str(output_path)),
        "preregistration_path": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "input_sha256": {
            "proposed_predictions": spec["proposed"]["predictions_sha256"],
            "proposed_per_example": spec["proposed"]["per_example_sha256"],
            "strongest_baseline_per_example": spec["strongest_completed_baseline"]["per_example_sha256"],
            "lead_metrics": spec["lead_metrics"]["sha256"],
        },
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(evidence_path, evidence)
    _append_search_log(
        {
            "logged_at_utc": _utc_now(),
            "study_id": protocol["study_id"],
            "dataset": spec["label"],
            "partition": "dev",
            "family": "greedy_reference_analysis",
            "candidate": "headroom_and_candidate_recall",
            "method": "preregistered_posthoc_analysis",
            "config_hash": PREREGISTRATION_SHA256,
            "run_attempt": "final",
            "dev_score": None,
            "dev_test_score": None,
            "status": "completed",
            "promoted": False,
            "reason": "diagnostic only; no promotion decision",
            "comparison_family_size": 2,
            "dev_test_accessed": False,
            "test_split_accessed": False,
        }
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("multinews", "govreport"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(analyze_dataset(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
