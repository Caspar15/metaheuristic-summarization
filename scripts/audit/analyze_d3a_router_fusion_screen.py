"""Analyze D3a pilot scores and apply the frozen full-dev finalist rule."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from scripts.audit.run_d3a_router_fusion_screen import (
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
STUDY_ROOT = REPO_ROOT / "runs_v2/d3a_router_fusion_screen_v1"
OUTPUT_PATH = STUDY_ROOT / "analysis/pilot_analysis.json"


def _protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    if sha256_file(str(path)) != PREREGISTRATION_SHA256:
        raise ValueError("D3a preregistration SHA-256 drifted")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("dev_test_access") != "none" or not value.get("test_split_prohibited"):
        raise ValueError("D3a analysis must remain outside protected splits")
    if len(value.get("candidates", [])) != 14:
        raise ValueError("D3a candidate matrix drifted")
    return value


def _per_example(
    path: Path, expected_ids: Sequence[str]
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    rows = [dict(row) for row in read_jsonl(str(path))]
    ids = [str(row.get("id")) for row in rows]
    if ids != list(expected_ids) or len(set(ids)) != len(ids):
        raise ValueError(f"D3a per-example order/identity drift: {_relative(path)}")
    scores = {metric: [] for metric in METRICS}
    for row in rows:
        values = []
        for metric in DEFAULT_METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError("D3a per-example score is non-finite")
            scores[metric].append(value)
            values.append(value)
        scores["macro_rouge"].append(fmean(values))
    return scores, {
        "path": _relative(path),
        "sha256": sha256_file(str(path)),
        "rows": len(rows),
    }


def select_full_dev_finalists(
    candidates: Sequence[Mapping[str, Any]], macro_scores: Mapping[str, float]
) -> list[str]:
    """Apply the exact preregistered family-winner/threshold/cost tie rule."""

    anchor = "R00_anchor"
    if anchor not in macro_scores:
        raise ValueError("D3a anchor score is missing")
    non_anchor = [row for row in candidates if row["id"] != anchor]
    families = sorted({str(row["family"]) for row in non_anchor})
    family_winners: list[str] = []
    for family in families:
        members = [row for row in non_anchor if row["family"] == family]
        winner = min(
            members,
            key=lambda row: (-float(macro_scores[str(row["id"])]), str(row["id"])),
        )
        family_winners.append(str(winner["id"]))
    selected = list(family_winners)
    anchor_score = float(macro_scores[anchor])
    eligible = [
        str(row["id"])
        for row in non_anchor
        if str(row["id"]) not in selected
        and float(macro_scores[str(row["id"])]) - anchor_score >= 0.002
    ]
    eligible.sort(key=lambda candidate_id: (-float(macro_scores[candidate_id]), candidate_id))
    selected.extend(eligible[: max(0, 4 - len(selected))])
    if len(selected) > 4:
        raise RuntimeError("D3a finalist cap was exceeded")
    return [anchor, *selected]


def analyze() -> dict[str, Any]:
    protocol = _protocol()
    raw_p: dict[str, float] = {}
    datasets: dict[str, Any] = {}
    all_endpoints: dict[str, dict[str, Any]] = {}
    endpoint_index = 0
    for slug, label in DATASET_LABELS.items():
        root = STUDY_ROOT / slug / "pilot"
        summary_path = root / "study_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            raise ValueError(f"D3a {label} pilot is incomplete")
        if summary.get("dev_test_accessed") is not False or summary.get("test_split_accessed") is not False:
            raise ValueError("D3a summary lacks protected-split guards")
        manifest_path = REPO_ROOT / protocol["datasets"][label]["pilot_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = [str(value) for value in manifest["selected_ids"]]
        candidate_scores: dict[str, dict[str, list[float]]] = {}
        artifacts: dict[str, Any] = {}
        macro_scores: dict[str, float] = {}
        for candidate in protocol["candidates"]:
            candidate_id = str(candidate["id"])
            candidate_root = root / candidate_id
            candidate_summary_path = candidate_root / "candidate_summary.json"
            candidate_summary = json.loads(candidate_summary_path.read_text(encoding="utf-8"))
            if candidate_summary.get("status") != "completed":
                raise ValueError(f"D3a candidate incomplete: {label}/{candidate_id}")
            if candidate_summary.get("dev_test_accessed") is not False or candidate_summary.get("test_split_accessed") is not False:
                raise ValueError("D3a candidate lacks protected-split guards")
            values, per_artifact = _per_example(candidate_root / "per_example.jsonl", ids)
            candidate_scores[candidate_id] = values
            macro_scores[candidate_id] = float(candidate_summary["metrics"]["macro_rouge"])
            artifacts[candidate_id] = {
                "summary_path": _relative(candidate_summary_path),
                "summary_sha256": sha256_file(str(candidate_summary_path)),
                "per_example": per_artifact,
            }
        anchor = candidate_scores["R00_anchor"]
        comparisons: dict[str, Any] = {}
        for candidate in protocol["candidates"][1:]:
            candidate_id = str(candidate["id"])
            endpoints: dict[str, Any] = {}
            for metric in METRICS:
                endpoint = paired_bootstrap_difference(
                    candidate_scores[candidate_id][metric],
                    anchor[metric],
                    n_resamples=10_000,
                    seed=20260809 + endpoint_index,
                )
                key = f"{label}:{candidate_id}:{metric}"
                raw_p[key] = float(endpoint["p_value_two_sided"])
                endpoint["p_value_selection_bonferroni_332"] = min(
                    1.0, raw_p[key] * 332
                )
                endpoints[metric] = endpoint
                all_endpoints[key] = endpoint
                endpoint_index += 1
            comparisons[candidate_id] = {
                "family": candidate["family"],
                "versus": "R00_anchor",
                "aggregate_macro": macro_scores[candidate_id],
                "aggregate_macro_delta": macro_scores[candidate_id] - macro_scores["R00_anchor"],
                "metrics": endpoints,
            }
        finalists = select_full_dev_finalists(protocol["candidates"], macro_scores)
        datasets[label] = {
            "pilot_rows": len(ids),
            "pilot_manifest_path": _relative(manifest_path),
            "pilot_manifest_sha256": sha256_file(str(manifest_path)),
            "anchor_macro": macro_scores["R00_anchor"],
            "macro_scores": macro_scores,
            "comparisons": comparisons,
            "full_dev_finalists": finalists,
            "artifact_registry": artifacts,
        }
    adjusted = holm_adjust(raw_p)
    for key, endpoint in all_endpoints.items():
        endpoint["p_value_holm_104"] = adjusted[key]
    result = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed_pilot_analysis_full_dev_finalists_selected",
        "study_id": protocol["study_id"],
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "paired_contract": {
            "resamples": 10_000,
            "base_seed": 20260809,
            "holm_endpoints": len(raw_p),
            "selection_opportunities": 83,
            "selection_endpoints": 332,
        },
        "datasets": datasets,
        "interpretation": "Pilot directs preregistered full-dev finalists only; it is not final superiority evidence.",
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_PATH, result)
    return result


if __name__ == "__main__":
    print(json.dumps(analyze()["status"]))
