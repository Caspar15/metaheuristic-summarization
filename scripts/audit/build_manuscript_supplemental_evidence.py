"""Build manuscript-only evidence from frozen predictions and score artifacts.

This script never runs a summarizer or evaluator. It derives output-length
statistics, a matched-row Multi-News sensitivity analysis, transparent
development configuration counts, and a deterministic development provenance
case from already frozen artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping

from scripts.audit.run_greedy_sensitivity import REPO_ROOT
from scripts.audit.run_length_contract_study import _relative, _utc_now, _write_json
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.utils.io import read_jsonl


OUTPUT_ROOT = REPO_ROOT / "runs_v2/manuscript_supplemental_analysis_v1"
RANDOM_SEEDS = (3407, 2024, 42, 1337, 2026, 20260810, 7, 17, 23, 101)
SYSTEMS = {
    "govreport": {
        "root": "runs_v2/govreport_final_test_v1",
        "rows": 973,
        "systems": {
            "Proposed": ("proposed",),
            "Lead": ("lead",),
            "Random (10 seeds)": tuple(f"random_seed_{seed}" for seed in RANDOM_SEEDS),
            "TextRank": ("textrank",),
            "LexRank": ("lexrank",),
            "PacSum-TFIDF": ("pacsum_tfidf_P07",),
            "PacSum-SBERT": ("pacsum_sbert_beta_0.5",),
            "SBERT centroid": ("sbert_centroid",),
            "SBERT+MMR": ("sbert_mmr_lambda_0.9",),
        },
    },
    "multinews": {
        "root": "runs_v2/multinews_final_test_v1",
        "rows": 5621,
        "systems": {
            "Proposed": ("proposed",),
            "Lead": ("lead",),
            "Random (10 seeds)": tuple(f"random_seed_{seed}" for seed in RANDOM_SEEDS),
            "TextRank": ("textrank",),
            "LexRank": ("lexrank",),
            "PacSum-TFIDF": ("pacsum_tfidf_P08",),
            "PacSum-SBERT": ("pacsum_sbert_P03",),
            "SBERT centroid": ("sbert_centroid",),
            "SBERT+MMR": ("sbert_mmr_lambda_0.7",),
        },
    },
}


def _length_statistics(paths: Iterable[Path]) -> dict[str, float | int]:
    words: list[int] = []
    sentences: list[int] = []
    feasible = 0
    for path in paths:
        for row in read_jsonl(str(path)):
            words.append(len(str(row.get("summary", "")).split()))
            sentences.append(len(row.get("selected_indices") or []))
            feasible += bool(row.get("feasible"))
    if not words:
        raise ValueError("length statistics received no prediction rows")
    return {
        "observations": len(words),
        "mean_words": fmean(words),
        "sd_words_population": pstdev(words),
        "mean_sentences": fmean(sentences),
        "sd_sentences_population": pstdev(sentences),
        "min_words": min(words),
        "max_words": max(words),
        "feasible_observations": feasible,
        "infeasible_observations": len(words) - feasible,
    }


def _all_length_statistics() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset, specification in SYSTEMS.items():
        root = REPO_ROOT / specification["root"]
        systems = {}
        for display, directories in specification["systems"].items():
            paths = [
                root / "predictions" / directory / "predictions.jsonl"
                for directory in directories
            ]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise ValueError(f"missing frozen prediction files: {missing}")
            systems[display] = _length_statistics(paths)
            systems[display]["aggregation"] = (
                "pooled document-seed observations"
                if len(paths) > 1
                else "documents"
            )
        output[dataset] = {
            "dataset_rows": specification["rows"],
            "systems": systems,
        }
    return output


def _rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = {str(row["id"]): row for row in read_jsonl(str(path))}
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def _multinews_sensitivity() -> dict[str, Any]:
    root = REPO_ROOT / "runs_v2/multinews_final_test_v1"
    proposed_predictions = root / "predictions/proposed/predictions.jsonl"
    infeasible_ids = [
        str(row["id"])
        for row in read_jsonl(str(proposed_predictions))
        if not bool(row.get("feasible"))
    ]
    if len(infeasible_ids) != 12:
        raise ValueError(f"expected 12 Proposed infeasible rows, got {len(infeasible_ids)}")
    proposed = _rows_by_id(root / "official/proposed/per_example.jsonl")
    comparator = _rows_by_id(
        root / "official/pacsum_tfidf_P08/per_example.jsonl"
    )
    if set(proposed) != set(comparator):
        raise ValueError("Multi-News official paired IDs drifted")
    excluded = set(infeasible_ids)
    matched_ids = [row_id for row_id in proposed if row_id not in excluded]
    if len(matched_ids) != 5609:
        raise ValueError(f"expected 5609 matched feasible IDs, got {len(matched_ids)}")

    metrics = ("rouge1", "rouge2", "rougeL", "macro_rouge")
    corpus: dict[str, Any] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    component_raw_p: dict[str, float] = {}
    for offset, metric in enumerate(metrics):
        left = [float(proposed[row_id][metric]) for row_id in matched_ids]
        right = [float(comparator[row_id][metric]) for row_id in matched_ids]
        corpus[metric] = {
            "proposed_per_example_mean": fmean(left),
            "pacsum_tfidf_per_example_mean": fmean(right),
        }
        result = paired_bootstrap_difference(
            left,
            right,
            n_resamples=100_000,
            seed=20260904 + offset,
        )
        comparisons[metric] = result
        if metric != "macro_rouge":
            component_raw_p[metric] = float(result["p_value_two_sided"])
    adjusted = holm_adjust(component_raw_p)
    for metric, value in adjusted.items():
        comparisons[metric]["p_value_holm_3"] = value
    return {
        "status": "post_hoc_fixed_output_sensitivity",
        "all_rows": 5621,
        "excluded_rows": 12,
        "matched_rows": len(matched_ids),
        "exclusion_rule": (
            "exclude the same Proposed-infeasible IDs from both systems; "
            "no output regeneration or method change"
        ),
        "infeasible_ids": infeasible_ids,
        "per_example_means": corpus,
        "proposed_minus_pacsum_tfidf": comparisons,
        "replaces_primary_all_row_result": False,
    }


def _configuration_budget() -> dict[str, Any]:
    rows = list(read_jsonl(str(REPO_ROOT / "runs_v2/search_log.jsonl")))
    baseline: dict[str, Any] = {}
    for dataset in ("GovReport", "Multi-News"):
        gate = [
            row
            for row in rows
            if row.get("dataset") == dataset
            and row.get("study_id") == "gate2-baseline-matrix-v1"
            and row.get("dev_score") is not None
        ]
        candidates = {str(row.get("candidate")) for row in gate}
        baseline[dataset] = {
            "Lead": 1,
            "Random": "10 fixed seeds; not selected by score",
            "TextRank": int("textrank" in candidates),
            "LexRank": int("lexrank" in candidates),
            "PacSum-TFIDF": sum(name.startswith("pacsum_tfidf_") for name in candidates),
            "PacSum-SBERT": sum(name.startswith("pacsum_sbert_") for name in candidates),
            "SBERT centroid": int("sbert_centroid" in candidates),
            "SBERT+MMR": sum(name.startswith("sbert_mmr_") for name in candidates),
        }

    proposed_studies = {
        "a1-length-contract-govreport-v1",
        "a1-length-contract-v1",
        "d1-capacity-matched-route-ablation-v1",
        "d1-govreport-section-guard-followup-v1",
        "d1-greedy-sensitivity-v1",
        "d1-three-route-capacity-followup-v1",
        "d2-selector-full-dev-v1",
        "d3a-router-fusion-full-dev-v1",
        "d3a-router-fusion-screen-v1",
        "d3b-cross-profile-combination-v1",
    }
    proposed: dict[str, Any] = {}
    for dataset in ("GovReport", "Multi-News"):
        eligible = [
            row
            for row in rows
            if row.get("dataset") == dataset
            and row.get("study_id") in proposed_studies
            and row.get("dev_score") is not None
        ]
        proposed[dataset] = {
            "scored_development_records": len(eligible),
            "unique_config_hashes": len(
                {str(row.get("config_hash")) for row in eligible}
            ),
            "held_out_dev_test_score_observations": sum(
                row.get("dev_test_score") is not None for row in eligible
            ),
            "scope_note": (
                "Includes prespecified length, capacity, route, selector, and "
                "combination studies that could inform the final profile; "
                "excludes E3, cost, oracle, and engineering checks."
            ),
        }
    return {
        "baseline_validation_candidates": baseline,
        "proposed_development_program": proposed,
        "interpretation": (
            "Counts document configuration selection, not model fine-tuning. "
            "No listed study accessed an official test split."
        ),
    }


def _provenance_case() -> dict[str, Any]:
    path = REPO_ROOT / (
        "runs_v2/d3b_cross_profile_combination_v1/govreport/dev/"
        "C01_combined_salience_route_weight/predictions.jsonl"
    )
    row = next(iter(read_jsonl(str(path))))
    records = {
        int(record["original_index"]): record
        for record in row.get("candidate_records") or []
    }
    selected = []
    for order, index in enumerate(row.get("selected_indices") or [], start=1):
        record = records[int(index)]
        selected.append({
            "selection_order": order,
            "sentence_id": record["sentence_id"],
            "original_index": int(index),
            "text": record["text"],
            "route_ranks": {
                route: details.get("rank")
                for route, details in record.get("route_scores", {}).items()
            },
            "selected_by_routes": record.get("selected_by_routes") or [],
            "route_agreement": record.get("route_agreement"),
            "fusion_rank": record.get("fused_rank"),
            "fusion_score": record.get("fusion_score"),
            "retention_reasons": record.get("inclusion_reasons") or [],
            "selected_in_summary": True,
        })
    selected_indices = {int(index) for index in row.get("selected_indices") or []}
    reservation_only = []
    for record in row.get("candidate_records") or []:
        reasons = list(record.get("inclusion_reasons") or [])
        index = int(record["original_index"])
        if index in selected_indices or not any(
            str(reason).startswith("reserve:") for reason in reasons
        ):
            continue
        reservation_only.append({
            "sentence_id": record["sentence_id"],
            "original_index": index,
            "text": record["text"],
            "route_ranks": {
                route: details.get("rank")
                for route, details in record.get("route_scores", {}).items()
            },
            "selected_by_routes": record.get("selected_by_routes") or [],
            "route_agreement": record.get("route_agreement"),
            "fusion_rank": record.get("fused_rank"),
            "retention_reasons": reasons,
            "selected_in_summary": False,
        })
        if len(reservation_only) == 5:
            break
    return {
        "selection_rule": (
            "first row in the frozen GovReport development prediction order; "
            "chosen without inspecting ROUGE or qualitative favorability"
        ),
        "source_artifact": _relative(path),
        "example_id": row["id"],
        "summary_words": len(str(row.get("summary", "")).split()),
        "selected_sentence_count": len(selected),
        "selected_sentences": selected,
        "first_five_reserved_not_selected": reservation_only,
        "test_split_accessed": False,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    case = _provenance_case()
    _write_json(OUTPUT_ROOT / "provenance_case.json", case)
    output = {
        "evidence_schema_version": "1.0",
        "measured_at": _utc_now(),
        "status": "completed",
        "study_id": "manuscript-supplemental-analysis-v1",
        "output_length": _all_length_statistics(),
        "multinews_feasible_row_sensitivity": _multinews_sensitivity(),
        "configuration_budget": _configuration_budget(),
        "provenance_case": {
            "path": _relative(OUTPUT_ROOT / "provenance_case.json"),
            "selection_rule": case["selection_rule"],
            "example_id": case["example_id"],
        },
        "new_test_inference": False,
        "post_score_tuning": False,
        "test_split_accessed": True,
        "test_access_reason": "read already frozen predictions and per-example scores only",
    }
    _write_json(OUTPUT_ROOT / "analysis.json", output)
    print(json.dumps({
        "status": output["status"],
        "output": _relative(OUTPUT_ROOT / "analysis.json"),
        "matched_rows": output["multinews_feasible_row_sensitivity"]["matched_rows"],
    }))


if __name__ == "__main__":
    main()
