"""Apply the preregistered D3a full-dev adversarial promotion gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from scripts.audit.run_d3a_router_fusion_full_dev import (
    DATASET_LABELS,
    PREREGISTRATION,
    PREREGISTRATION_SHA256,
)
from scripts.audit.run_greedy_sensitivity import REPO_ROOT
from scripts.audit.run_length_contract_study import _relative, _utc_now, _write_json
from src.data.policy import sha256_file
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.eval.rouge import DEFAULT_METRICS
from src.utils.io import read_jsonl


METRICS = (*DEFAULT_METRICS, "macro_rouge")
STUDY_ROOT = REPO_ROOT / "runs_v2/d3a_router_fusion_full_dev_v1"
OUTPUT_PATH = STUDY_ROOT / "analysis/paired_summary.json"


def _protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    if sha256_file(str(path)) != PREREGISTRATION_SHA256:
        raise ValueError("D3a full-dev preregistration drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("partition") != "dev" or value.get("dev_test_access") != "none":
        raise ValueError("D3a full-dev analysis must remain on dev")
    if value.get("test_split_prohibited") is not True:
        raise ValueError("D3a full-dev analysis must prohibit test")
    return value


def _scores(
    path: Path, expected_ids: Sequence[str] | None = None
) -> tuple[list[str], dict[str, list[float]], dict[str, Any]]:
    rows = [dict(row) for row in read_jsonl(str(path))]
    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)) or not all(ids):
        raise ValueError(f"duplicate/invalid IDs in {_relative(path)}")
    if expected_ids is not None and ids != list(expected_ids):
        raise ValueError(f"row order mismatch in {_relative(path)}")
    values = {metric: [] for metric in METRICS}
    for row in rows:
        component = []
        for metric in DEFAULT_METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError("non-finite D3a full-dev score")
            values[metric].append(value)
            component.append(value)
        values["macro_rouge"].append(fmean(component))
    return ids, values, {
        "path": _relative(path),
        "sha256": sha256_file(str(path)),
        "rows": len(rows),
    }


def promotion_decision(
    comparison: Mapping[str, Mapping[str, float]], *,
    macro_point_delta: float,
) -> dict[str, Any]:
    macro = comparison["macro_rouge"]
    components = [comparison[name] for name in DEFAULT_METRICS]
    checks = {
        "macro_point_positive": macro_point_delta > 0.0,
        "macro_ci_lower_positive": float(macro["ci_lower"]) > 0.0,
        "macro_holm_28_le_005": float(macro["p_value_holm_28"]) <= 0.05,
        "macro_selection_bonferroni_332_le_005": float(
            macro["p_value_selection_bonferroni_332"]
        ) <= 0.05,
        "no_component_ci_entirely_negative": all(
            float(endpoint["ci_upper"]) >= 0.0 for endpoint in components
        ),
    }
    return {"eligible": all(checks.values()), "checks": checks}


def analyze() -> dict[str, Any]:
    protocol = _protocol()
    raw_p: dict[str, float] = {}
    formal_endpoints: dict[str, dict[str, Any]] = {}
    datasets: dict[str, Any] = {}
    endpoint_index = 0
    for slug, label in DATASET_LABELS.items():
        registration = protocol["datasets"][label]
        root = STUDY_ROOT / slug / "dev"
        study = json.loads((root / "study_summary.json").read_text(encoding="utf-8"))
        if study.get("status") != "completed":
            raise ValueError(f"D3a full-dev study incomplete: {label}")
        if study.get("dev_test_accessed") is not False or study.get("test_split_accessed") is not False:
            raise ValueError("D3a full-dev study lacks protected-split guards")
        anchor_result = study["candidates"]["R00_anchor"]
        anchor_path = REPO_ROOT / anchor_result["per_example_path"]
        ids, anchor, anchor_artifact = _scores(anchor_path)
        if len(ids) != int(registration["rows"]):
            raise ValueError("D3a full-dev anchor row count drifted")
        baseline_path = REPO_ROOT / registration["adversarial_per_example"]
        if sha256_file(str(baseline_path)) != registration["adversarial_per_example_sha256"]:
            raise ValueError("D3a adversarial per-example artifact drifted")
        _, baseline, baseline_artifact = _scores(baseline_path, ids)
        candidates: dict[str, Any] = {}
        macro_scores: dict[str, float] = {
            "R00_anchor": float(anchor_result["metrics"]["macro_rouge"])
        }
        for candidate_id in registration["finalists"][1:]:
            result = study["candidates"][candidate_id]
            if result.get("status") != "completed":
                raise ValueError(f"D3a full-dev candidate incomplete: {candidate_id}")
            path = REPO_ROOT / result["per_example_path"]
            _, values, artifact = _scores(path, ids)
            macro_scores[candidate_id] = float(result["metrics"]["macro_rouge"])
            vs_anchor: dict[str, Any] = {}
            vs_baseline: dict[str, Any] = {}
            for metric in METRICS:
                vs_anchor[metric] = paired_bootstrap_difference(
                    values[metric], anchor[metric],
                    n_resamples=10_000, seed=20270809 + endpoint_index,
                )
                formal = paired_bootstrap_difference(
                    values[metric], baseline[metric],
                    n_resamples=10_000, seed=20260809 + endpoint_index,
                )
                key = f"{label}:{candidate_id}:{metric}"
                raw_p[key] = float(formal["p_value_two_sided"])
                formal["p_value_selection_bonferroni_332"] = min(
                    1.0, raw_p[key] * 332
                )
                formal_endpoints[key] = formal
                vs_baseline[metric] = formal
                endpoint_index += 1
            candidates[candidate_id] = {
                "macro": macro_scores[candidate_id],
                "macro_delta_vs_anchor": macro_scores[candidate_id] - macro_scores["R00_anchor"],
                "per_example": artifact,
                "versus_anchor_descriptive": vs_anchor,
                "versus_adversarial_formal": vs_baseline,
            }
        winner = min(
            registration["finalists"],
            key=lambda candidate_id: (-macro_scores[candidate_id], candidate_id),
        )
        datasets[label] = {
            "rows": len(ids),
            "anchor": {"macro": macro_scores["R00_anchor"], "per_example": anchor_artifact},
            "adversarial_baseline": {
                "label": registration["adversarial_baseline"],
                "per_example": baseline_artifact,
                "macro": fmean(baseline["macro_rouge"]),
            },
            "candidates": candidates,
            "winner": winner,
            "winner_macro": macro_scores[winner],
        }
    adjusted = holm_adjust(raw_p)
    for key, endpoint in formal_endpoints.items():
        endpoint["p_value_holm_28"] = adjusted[key]
    for label, result in datasets.items():
        winner = result["winner"]
        if winner == "R00_anchor":
            result["promotion"] = {
                "eligible": False,
                "reason": "anchor remained winner; no D3a finalist can promote",
            }
            continue
        comparison = result["candidates"][winner]["versus_adversarial_formal"]
        point_delta = result["winner_macro"] - result["adversarial_baseline"]["macro"]
        result["winner_macro_delta_vs_adversarial"] = point_delta
        result["promotion"] = promotion_decision(
            comparison, macro_point_delta=point_delta
        )
    all_profiles_eligible = all(
        bool(result["promotion"]["eligible"]) for result in datasets.values()
    )
    output = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed_no_protected_split_promotion" if not all_profiles_eligible else "completed_freeze_recommendation_required",
        "study_id": protocol["study_id"],
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "paired_contract": {
            "resamples": 10_000,
            "base_seed_formal": 20260809,
            "holm_endpoints": len(raw_p),
            "selection_opportunities": 83,
            "selection_endpoints": 332,
            "finite_bootstrap_minimum_two_sided_p": 2 / 10001,
            "minimum_possible_bonferroni_332": (2 / 10001) * 332,
        },
        "datasets": datasets,
        "all_profiles_eligible": all_profiles_eligible,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_PATH, output)
    return output


if __name__ == "__main__":
    print(json.dumps(analyze()["status"]))
