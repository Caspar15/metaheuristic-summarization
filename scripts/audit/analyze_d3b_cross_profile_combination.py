"""Apply the preregistered D3b final-combination promotion gate."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from scripts.audit.run_d3b_cross_profile_combination import (
    DATASET_LABELS,
    PREREGISTRATION,
    PREREGISTRATION_SHA256,
    _protocol,
)
from scripts.audit.run_greedy_sensitivity import REPO_ROOT
from scripts.audit.run_length_contract_study import _relative, _utc_now, _write_json
from src.data.policy import sha256_file
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.eval.rouge import DEFAULT_METRICS
from src.utils.io import read_jsonl


METRICS = (*DEFAULT_METRICS, "macro_rouge")
STUDY_ROOT = REPO_ROOT / "runs_v2/d3b_cross_profile_combination_v1"
OUTPUT_PATH = STUDY_ROOT / "analysis/paired_summary.json"
FORMAL_RESAMPLES = 100_000
DESCRIPTIVE_RESAMPLES = 10_000
FORMAL_SEED = 20260809
DESCRIPTIVE_SEED = 20270809
HOLM_ENDPOINTS = 8
SELECTION_ENDPOINTS = 340


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
        components = []
        for metric in DEFAULT_METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError("non-finite D3b score")
            values[metric].append(value)
            components.append(value)
        values["macro_rouge"].append(fmean(components))
    return ids, values, {
        "path": _relative(path),
        "sha256": sha256_file(str(path)),
        "rows": len(rows),
    }


def promotion_decision(
    comparison: Mapping[str, Mapping[str, float]], *, macro_point_delta: float
) -> dict[str, Any]:
    macro = comparison["macro_rouge"]
    components = [comparison[name] for name in DEFAULT_METRICS]
    checks = {
        "macro_point_positive": macro_point_delta > 0.0,
        "macro_ci_lower_positive": float(macro["ci_lower"]) > 0.0,
        "macro_holm_8_le_005": float(macro["p_value_holm_8"]) <= 0.05,
        "macro_selection_bonferroni_340_le_005": float(
            macro["p_value_selection_bonferroni_340"]
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
        study_path = root / "study_summary.json"
        study = json.loads(study_path.read_text(encoding="utf-8"))
        if study.get("status") != "completed":
            raise ValueError(f"D3b study incomplete: {label}")
        if (
            study.get("dev_test_accessed") is not False
            or study.get("test_split_accessed") is not False
        ):
            raise ValueError("D3b study lacks protected-split guards")
        result = study["combination"]
        if result.get("status") != "completed":
            raise ValueError(f"D3b candidate incomplete: {label}")
        candidate_path = REPO_ROOT / result["per_example_path"]
        ids, candidate, candidate_artifact = _scores(candidate_path)
        if len(ids) != int(registration["rows"]):
            raise ValueError("D3b candidate row count drifted")

        anchor_summary_path = REPO_ROOT / registration["anchor_summary"]
        anchor_summary = json.loads(anchor_summary_path.read_text(encoding="utf-8"))
        anchor_path = REPO_ROOT / anchor_summary["per_example_path"]
        _, anchor, anchor_artifact = _scores(anchor_path, ids)

        baseline_path = REPO_ROOT / registration["adversarial_per_example"]
        if sha256_file(str(baseline_path)) != registration["adversarial_per_example_sha256"]:
            raise ValueError("D3b adversarial per-example artifact drifted")
        _, baseline, baseline_artifact = _scores(baseline_path, ids)

        versus_anchor: dict[str, Any] = {}
        versus_baseline: dict[str, Any] = {}
        for metric in METRICS:
            versus_anchor[metric] = paired_bootstrap_difference(
                candidate[metric],
                anchor[metric],
                n_resamples=DESCRIPTIVE_RESAMPLES,
                seed=DESCRIPTIVE_SEED + endpoint_index,
            )
            formal = paired_bootstrap_difference(
                candidate[metric],
                baseline[metric],
                n_resamples=FORMAL_RESAMPLES,
                seed=FORMAL_SEED + endpoint_index,
            )
            key = f"{label}:{metric}"
            raw_p[key] = float(formal["p_value_two_sided"])
            formal["p_value_selection_bonferroni_340"] = min(
                1.0, raw_p[key] * SELECTION_ENDPOINTS
            )
            formal_endpoints[key] = formal
            versus_baseline[metric] = formal
            endpoint_index += 1

        candidate_macro = float(result["metrics"]["macro_rouge"])
        anchor_macro = float(anchor_summary["metrics"]["macro_rouge"])
        baseline_macro = fmean(baseline["macro_rouge"])
        datasets[label] = {
            "rows": len(ids),
            "combination": {
                "macro": candidate_macro,
                "macro_delta_vs_D3a_anchor": candidate_macro - anchor_macro,
                "per_example": candidate_artifact,
            },
            "D3a_anchor": {
                "candidate": registration["anchor_candidate"],
                "macro": anchor_macro,
                "per_example": anchor_artifact,
            },
            "adversarial_baseline": {
                "label": registration["adversarial_baseline"],
                "macro": baseline_macro,
                "per_example": baseline_artifact,
            },
            "macro_delta_vs_adversarial": candidate_macro - baseline_macro,
            "versus_D3a_anchor_descriptive": versus_anchor,
            "versus_adversarial_formal": versus_baseline,
        }

    if len(raw_p) != HOLM_ENDPOINTS:
        raise ValueError("D3b formal Holm family size drifted")
    adjusted = holm_adjust(raw_p)
    for key, endpoint in formal_endpoints.items():
        endpoint["p_value_holm_8"] = adjusted[key]
    for result in datasets.values():
        result["promotion"] = promotion_decision(
            result["versus_adversarial_formal"],
            macro_point_delta=float(result["macro_delta_vs_adversarial"]),
        )

    all_profiles_eligible = all(
        bool(result["promotion"]["eligible"]) for result in datasets.values()
    )
    output = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": (
            "completed_freeze_recommendation_required"
            if all_profiles_eligible
            else "completed_repositioning_required"
        ),
        "study_id": protocol["study_id"],
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "paired_contract": {
            "formal_resamples": FORMAL_RESAMPLES,
            "descriptive_resamples": DESCRIPTIVE_RESAMPLES,
            "base_seed_formal": FORMAL_SEED,
            "holm_endpoints": len(raw_p),
            "selection_opportunities": 85,
            "selection_endpoints": SELECTION_ENDPOINTS,
            "finite_bootstrap_minimum_two_sided_p": 2 / (FORMAL_RESAMPLES + 1),
            "minimum_possible_bonferroni_340": (
                2 / (FORMAL_RESAMPLES + 1)
            ) * SELECTION_ENDPOINTS,
        },
        "datasets": datasets,
        "all_profiles_eligible": all_profiles_eligible,
        "decision": (
            "stop before dev-test/test and write freeze recommendation"
            if all_profiles_eligible
            else "stop search and write repositioning recommendation; no dev-test/test"
        ),
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_PATH, output)
    return output


if __name__ == "__main__":
    print(json.dumps(analyze()["status"]))
