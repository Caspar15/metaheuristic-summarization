"""Analyze the preregistered D2 selector-only screen on frozen dev.

This script consumes existing per-example score artifacts.  It has no split
argument and no code path to dev-test or a dataset test split.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import platform
from statistics import fmean
from typing import Any, Mapping, Sequence

import yaml

from scripts.audit.run_greedy_sensitivity import REPO_ROOT
from scripts.audit.run_length_contract_study import (
    _append_search_log,
    _canonical_sha256,
    _git_commit,
    _relative,
    _utc_now,
    _write_json,
)
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.eval.rouge import DEFAULT_METRICS
from src.utils.io import read_jsonl


PREREGISTRATION = "configs/preregistrations/d2_selector_full_dev_v1.json"
PREREGISTRATION_SHA256 = (
    "98d0093a7acb22cfbe43aa35bd5aed33fb7fbab85639ac1c920ee15ba8b11382"
)
OUTPUT_ROOT = REPO_ROOT / "runs_v2/d2_selector_full_dev_v1/analysis"
STUDY_ROOT = REPO_ROOT / "runs_v2/d2_selector_full_dev_v1"
METRICS = (*DEFAULT_METRICS, "macro_rouge")
DATASET_SLUGS = {"Multi-News": "multinews", "GovReport": "govreport"}
INPUT_PROFILES = {
    "Multi-News": {"input_mode": "multi_document", "output_mode": "multi_sentence"},
    "GovReport": {"input_mode": "single_document", "output_mode": "multi_sentence"},
}


def _load_protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    actual = sha256_file(str(path))
    if actual != PREREGISTRATION_SHA256:
        raise ValueError(f"D2 selector preregistration drifted: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("study_id") != "d2-selector-full-dev-v1":
        raise ValueError("D2 study identity drifted")
    if protocol.get("partition") != "dev" or protocol.get("dev_test_access") != "none":
        raise ValueError("D2 analysis must remain bound to frozen dev")
    if protocol.get("test_split_prohibited") is not True:
        raise ValueError("D2 analysis must prohibit test")
    candidates = protocol.get("candidates", [])
    if len(candidates) != 14 or sum(bool(c.get("reuse_base_run")) for c in candidates) != 1:
        raise ValueError("D2 candidate matrix drifted")
    measurement = protocol.get("measurement", {})
    if measurement.get("paired") != "10,000 paired bootstrap resamples, seed 20260809":
        raise ValueError("D2 paired-bootstrap contract drifted")
    if not str(measurement.get("family", "")).startswith("2 datasets x 13"):
        raise ValueError("D2 paired family drifted")
    if "57 proposed search operations" not in str(measurement.get("selection_opportunities")):
        raise ValueError("D2 search-opportunity count drifted")
    return protocol


def _frozen_ids(dataset_spec: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    config_path = REPO_ROOT / str(dataset_spec["base_config"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = REPO_ROOT / str(config["experiment_partition"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_optimization_scores":
        raise ValueError("D2 partition manifest is not frozen before optimization scores")
    dev = manifest.get("partitions", {}).get("dev", {})
    ids = dev.get("selected_ids")
    if not isinstance(ids, list) or not all(isinstance(value, str) and value for value in ids):
        raise ValueError("D2 partition manifest has invalid dev IDs")
    if len(ids) != int(dataset_spec["rows"]) or dev.get("rows") != len(ids):
        raise ValueError("D2 partition row count drifted")
    digest = selected_ids_sha256(ids)
    if dev.get("selected_ids_sha256") != digest:
        raise ValueError("D2 partition selected-ID digest drifted")
    return ids, {
        "path": _relative(manifest_path),
        "sha256": sha256_file(str(manifest_path)),
        "selected_ids_sha256": digest,
        "rows": len(ids),
    }


def _load_per_example(
    path: Path, expected_ids: Sequence[str]
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    rows = [dict(row) for row in read_jsonl(str(path))]
    actual_ids = [str(row.get("id")) for row in rows]
    if actual_ids != list(expected_ids):
        raise ValueError(f"{_relative(path)} is not in exact frozen-dev order")
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError(f"{_relative(path)} contains duplicate IDs")
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for position, row in enumerate(rows):
        row_values: list[float] = []
        for metric in DEFAULT_METRICS:
            if metric not in row:
                raise ValueError(f"{_relative(path)} row {position} lacks {metric}")
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"{_relative(path)} row {position} has non-finite {metric}")
            values[metric].append(value)
            row_values.append(value)
        values["macro_rouge"].append(fmean(row_values))
    evidence_path = path.parent / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("dev_test_accessed") is not False:
        raise ValueError(f"{_relative(evidence_path)} lacks a false dev-test guard")
    if evidence.get("test_split_accessed") is not False:
        raise ValueError(f"{_relative(evidence_path)} lacks a false test guard")
    return values, {
        "path": _relative(path),
        "sha256": sha256_file(str(path)),
        "rows": len(rows),
        "evidence_path": _relative(evidence_path),
        "evidence_sha256": sha256_file(str(evidence_path)),
    }


def _candidate_run_dir(
    protocol: Mapping[str, Any], dataset: str, candidate: Mapping[str, Any]
) -> Path:
    if candidate.get("reuse_base_run"):
        return REPO_ROOT / str(protocol["datasets"][dataset]["base_run"])
    slug = DATASET_SLUGS[dataset]
    return STUDY_ROOT / slug / "dev" / str(candidate["id"]) / str(candidate["method"]) / "run"


def _paired_outcomes(
    candidate_scores: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    anchor_id: str,
    n_resamples: int,
    seed: int,
    selection_opportunities: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    anchor = candidate_scores[anchor_id]
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    test_index = 0
    for candidate_id, scores in candidate_scores.items():
        if candidate_id == anchor_id:
            continue
        endpoints: dict[str, Any] = {}
        for metric in METRICS:
            endpoint = paired_bootstrap_difference(
                scores[metric], anchor[metric], n_resamples=n_resamples,
                seed=seed + test_index,
            )
            key = f"{candidate_id}:{metric}"
            raw_p[key] = float(endpoint["p_value_two_sided"])
            endpoint["p_value_selection_bonferroni"] = min(
                1.0, raw_p[key] * selection_opportunities
            )
            endpoints[metric] = endpoint
            test_index += 1
        comparisons[candidate_id] = {"versus": anchor_id, "metrics": endpoints}
    return comparisons, raw_p


def _gate2_adversarial_gap(dataset_slug: str, best_minus_anchor: float) -> dict[str, Any]:
    path = REPO_ROOT / "runs_v2/gate2_paired_finalists_v1/summary.json"
    gate2 = json.loads(path.read_text(encoding="utf-8"))
    if gate2.get("dev_test_accessed") is not False or gate2.get("test_split_accessed") is not False:
        raise ValueError("Gate 2 paired artifact accessed a protected split")
    result = gate2["datasets"][dataset_slug]
    label = result["adversarial_baseline"]
    anchor_gap = float(result["comparisons"][label]["metrics"]["macro_rouge"]["mean_difference"])
    return {
        "baseline": label,
        "anchor_minus_baseline_macro": anchor_gap,
        "best_d2_minus_baseline_macro": anchor_gap + best_minus_anchor,
        "source": _relative(path),
        "source_sha256": sha256_file(str(path)),
    }


def run() -> dict[str, Any]:
    protocol = _load_protocol()
    if OUTPUT_ROOT.exists():
        raise ValueError(f"refusing to overwrite D2 paired analysis: {OUTPUT_ROOT}")
    candidates = protocol["candidates"]
    anchor_id = str(next(item["id"] for item in candidates if item.get("reuse_base_run")))
    n_resamples = 10_000
    base_seed = 20_260_809
    # 57 searched proposed configurations, each carrying four reported endpoints.
    selection_opportunities = 57 * len(METRICS)
    all_raw_p: dict[str, float] = {}
    dataset_results: dict[str, Any] = {}
    deterministic_macro: dict[str, dict[str, float]] = {}
    for dataset_index, (dataset, dataset_spec) in enumerate(protocol["datasets"].items()):
        slug = DATASET_SLUGS[dataset]
        study_summary_path = STUDY_ROOT / slug / "dev" / "study_summary.json"
        study_summary = json.loads(study_summary_path.read_text(encoding="utf-8"))
        if study_summary.get("status") != "completed":
            raise ValueError(f"{dataset} D2 screen is incomplete")
        if study_summary.get("dev_test_accessed") is not False or study_summary.get("test_split_accessed") is not False:
            raise ValueError(f"{dataset} D2 screen accessed a protected split")
        expected_ids, manifest_provenance = _frozen_ids(dataset_spec)
        scores: dict[str, dict[str, list[float]]] = {}
        provenance: dict[str, Any] = {}
        methods: dict[str, str] = {}
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            run_dir = _candidate_run_dir(protocol, dataset, candidate)
            candidate_values, candidate_provenance = _load_per_example(
                run_dir / "per_example.jsonl", expected_ids
            )
            scores[candidate_id] = candidate_values
            provenance[candidate_id] = candidate_provenance
            methods[candidate_id] = str(candidate["method"])
        comparisons, raw_p = _paired_outcomes(
            scores, anchor_id=anchor_id, n_resamples=n_resamples,
            seed=base_seed + dataset_index * 1_000,
            selection_opportunities=selection_opportunities,
        )
        all_raw_p.update({f"{slug}:{key}": value for key, value in raw_p.items()})
        macro_means = {candidate_id: fmean(values["macro_rouge"]) for candidate_id, values in scores.items()}
        deterministic = {
            candidate_id: value for candidate_id, value in macro_means.items()
            if methods[candidate_id] != "nsga2"
        }
        deterministic_macro[dataset] = deterministic
        winner = max(deterministic, key=deterministic.__getitem__)
        best_minus_anchor = deterministic[winner] - macro_means[anchor_id]
        dataset_results[slug] = {
            "label": dataset,
            "input_profile": INPUT_PROFILES[dataset],
            "rows": len(expected_ids),
            "manifest": manifest_provenance,
            "study_summary": {
                "path": _relative(study_summary_path),
                "sha256": sha256_file(str(study_summary_path)),
            },
            "candidate_artifacts": provenance,
            "macro_means": macro_means,
            "best_deterministic_candidate": winner,
            "best_deterministic_minus_anchor_macro": best_minus_anchor,
            "adversarial_gate": _gate2_adversarial_gap(slug, best_minus_anchor),
            "comparisons_vs_anchor": comparisons,
        }
    if len(all_raw_p) != 104:
        raise ValueError(f"observed D2 paired family size drifted: {len(all_raw_p)}")
    adjusted = holm_adjust(all_raw_p)
    for slug, result in dataset_results.items():
        for candidate_id, comparison in result["comparisons_vs_anchor"].items():
            for metric, endpoint in comparison["metrics"].items():
                key = f"{slug}:{candidate_id}:{metric}"
                endpoint["p_value_holm"] = adjusted[key]
                endpoint["descriptive_endpoint_win"] = bool(
                    endpoint["mean_difference"] > 0.0
                    and endpoint["ci_lower"] > 0.0
                    and endpoint["p_value_holm"] < 0.05
                )
                endpoint["selection_aware_win"] = bool(
                    endpoint["descriptive_endpoint_win"]
                    and endpoint["p_value_selection_bonferroni"] < 0.05
                )
    shared_candidates = set.intersection(*(set(values) for values in deterministic_macro.values()))
    qualifying_shared = []
    for candidate_id in sorted(shared_candidates):
        deficits = {
            dataset: max(values.values()) - values[candidate_id]
            for dataset, values in deterministic_macro.items()
        }
        if all(value <= 0.001 for value in deficits.values()):
            qualifying_shared.append({"candidate": candidate_id, "macro_deficits": deficits})
    profile_policy = {
        json.dumps(INPUT_PROFILES[result["label"]], sort_keys=True): result["best_deterministic_candidate"]
        for result in dataset_results.values()
    }
    promotion_eligible = all(
        result["adversarial_gate"]["best_d2_minus_baseline_macro"] > 0.0
        for result in dataset_results.values()
    )
    summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed_dev_analysis_no_promotion" if not promotion_eligible else "completed_dev_analysis",
        "study_id": protocol["study_id"],
        "implementation_commit": _git_commit(),
        "preregistration_path": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "scientific_config_sha256": _canonical_sha256(protocol),
        "partition": "dev",
        "dev_test_accessed": False,
        "test_split_accessed": False,
        "bootstrap": {"resamples": n_resamples, "seed": base_seed, "confidence": 0.95},
        "multiplicity": {
            "holm_family_size": 104,
            "prior_and_current_proposed_search_operations": 57,
            "reported_endpoints_per_operation": len(METRICS),
            "selection_aware_bonferroni_opportunities": selection_opportunities,
        },
        "datasets": dataset_results,
        "cross_dataset_selection": {
            "shared_selector_tolerance": 0.001,
            "qualifying_shared_selectors": qualifying_shared,
            "decision": "shared_selector" if qualifying_shared else "task_profile_specific_selector",
            "task_profile_policy": profile_policy if not qualifying_shared else None,
        },
        "promotion_eligible": promotion_eligible,
        "interpretation": (
            "Frozen-dev paired analysis only. Aggregate failure against either adversarial "
            "baseline blocks dev-test promotion; test remains prohibited."
        ),
        "environment": {"python": platform.python_version()},
    }
    _write_json(OUTPUT_ROOT / "paired_summary.json", summary)
    _append_search_log({
        "logged_at_utc": _utc_now(),
        "study_id": protocol["study_id"],
        "dataset": "Multi-News+GovReport",
        "partition": "dev",
        "family": "selector_full_dev_paired",
        "candidate": "13_non_anchor_candidates_vs_anchor",
        "method": "paired_bootstrap_existing_per_example",
        "config_path": PREREGISTRATION,
        "config_hash": _canonical_sha256(protocol),
        "run_attempt": "final",
        "dev_score": None,
        "dev_test_score": None,
        "status": "completed",
        "promoted": promotion_eligible,
        "reason": "aggregate adversarial-baseline gate" if not promotion_eligible else "paired gate required next",
        "comparison_family_size": 104,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    })
    return summary


def main() -> None:
    summary = run()
    compact = {
        slug: {
            "winner": result["best_deterministic_candidate"],
            "winner_minus_anchor": result["best_deterministic_minus_anchor_macro"],
            "winner_minus_adversarial_baseline": result["adversarial_gate"]["best_d2_minus_baseline_macro"],
        }
        for slug, result in summary["datasets"].items()
    }
    compact["cross_dataset_selection"] = summary["cross_dataset_selection"]
    compact["promotion_eligible"] = summary["promotion_eligible"]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
