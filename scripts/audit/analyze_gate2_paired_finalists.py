"""Run the preregistered Gate 2 paired finalist diagnostic on frozen dev.

The script consumes already-scored per-example artifacts. It has no split
argument and no path to dev-test or a dataset test split.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
import platform
from statistics import fmean
from typing import Any, Mapping, Sequence

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


PREREGISTRATION = "configs/preregistrations/gate2_paired_finalists_v1.json"
PREREGISTRATION_SHA256 = (
    "8d8753f6d5e9a4ae65e87755bb579586e6e22c81d8c230175f34682ddfa08005"
)
OUTPUT_ROOT = REPO_ROOT / "runs_v2/gate2_paired_finalists_v1"
METRICS = (*DEFAULT_METRICS, "macro_rouge")


def _load_protocol() -> dict[str, Any]:
    path = REPO_ROOT / PREREGISTRATION
    actual = sha256_file(str(path))
    if actual != PREREGISTRATION_SHA256:
        raise ValueError(f"Gate 2 paired preregistration drifted: {actual}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_any_gate2_paired_resampling_outcome":
        raise ValueError("Gate 2 paired protocol was not frozen before outcomes")
    if protocol.get("partition") != "dev":
        raise ValueError("Gate 2 paired analysis must use frozen dev")
    if protocol.get("dev_test_accessed") is not False:
        raise ValueError("Gate 2 paired analysis must not access dev-test")
    if protocol.get("test_split_prohibited") is not True:
        raise ValueError("Gate 2 paired analysis must prohibit test")
    if tuple(protocol.get("metrics", ())) != METRICS:
        raise ValueError("Gate 2 paired metric contract drifted")
    bootstrap = protocol.get("bootstrap", {})
    if bootstrap.get("resamples") != 10_000 or bootstrap.get("seed") != 20_260_809:
        raise ValueError("Gate 2 paired bootstrap contract drifted")
    multiplicity = protocol.get("multiplicity", {})
    if multiplicity.get("holm_family_size") != 64:
        raise ValueError("Gate 2 paired Holm family drifted")
    if multiplicity.get("selection_aware_bonferroni_opportunities") != 12_896:
        raise ValueError("Gate 2 selection-opportunity count drifted")
    datasets = protocol.get("datasets", {})
    if set(datasets) != {"multinews", "govreport"}:
        raise ValueError("Gate 2 paired dataset matrix drifted")
    if any(len(spec.get("baselines", {})) != 8 for spec in datasets.values()):
        raise ValueError("Gate 2 paired finalist count drifted")
    return protocol


def _frozen_ids(spec: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    path = REPO_ROOT / str(spec["manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_optimization_scores":
        raise ValueError("paired manifest is not frozen before optimization scores")
    dev = manifest.get("partitions", {}).get("dev", {})
    ids = dev.get("selected_ids")
    if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids):
        raise ValueError("paired manifest has invalid dev IDs")
    if len(ids) != int(spec["rows"]) or dev.get("rows") != len(ids):
        raise ValueError("paired manifest row count drifted")
    digest = selected_ids_sha256(ids)
    if dev.get("selected_ids_sha256") != digest:
        raise ValueError("paired manifest selected-ID digest drifted")
    return ids, {
        "path": _relative(path),
        "sha256": sha256_file(str(path)),
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
        metric_values = []
        for metric in DEFAULT_METRICS:
            if metric not in row:
                raise ValueError(f"{_relative(path)} row {position} lacks {metric}")
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"{_relative(path)} row {position} has non-finite {metric}")
            values[metric].append(value)
            metric_values.append(value)
        values["macro_rouge"].append(fmean(metric_values))
    sibling_evidence = path.parent / "evidence.json"
    evidence_provenance: dict[str, Any] | None = None
    if sibling_evidence.exists():
        evidence = json.loads(sibling_evidence.read_text(encoding="utf-8"))
        if evidence.get("dev_test_accessed") is True or evidence.get("test_split_accessed") is True:
            raise ValueError(f"{_relative(sibling_evidence)} accessed a protected split")
        expected_sha = evidence.get("per_example_sha256")
        actual_sha = sha256_file(str(path))
        if expected_sha is not None and expected_sha != actual_sha:
            raise ValueError(f"{_relative(path)} SHA differs from sibling evidence")
        evidence_provenance = {
            "path": _relative(sibling_evidence),
            "sha256": sha256_file(str(sibling_evidence)),
        }
    return values, {
        "path": _relative(path),
        "sha256": sha256_file(str(path)),
        "rows": len(rows),
        "evidence": evidence_provenance,
    }


def _paired_outcomes(
    proposed: Mapping[str, Sequence[float]],
    baselines: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    n_resamples: int,
    seed: int,
    selection_opportunities: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    test_index = 0
    for baseline_label, baseline in baselines.items():
        metric_results: dict[str, Any] = {}
        for metric in METRICS:
            result = paired_bootstrap_difference(
                proposed[metric], baseline[metric], n_resamples=n_resamples,
                seed=seed + test_index,
            )
            key = f"{baseline_label}:{metric}"
            raw_p[key] = float(result["p_value_two_sided"])
            result["p_value_selection_bonferroni"] = min(
                1.0, raw_p[key] * selection_opportunities
            )
            metric_results[metric] = result
            test_index += 1
        comparisons[baseline_label] = {"metrics": metric_results}
    return comparisons, raw_p


def run() -> dict[str, Any]:
    protocol = _load_protocol()
    if OUTPUT_ROOT.exists():
        raise ValueError(f"refusing to overwrite Gate 2 paired analysis: {OUTPUT_ROOT}")
    n_resamples = int(protocol["bootstrap"]["resamples"])
    base_seed = int(protocol["bootstrap"]["seed"])
    opportunities = int(protocol["multiplicity"]["selection_aware_bonferroni_opportunities"])
    dataset_results: dict[str, Any] = {}
    all_raw_p: dict[str, float] = {}
    for dataset_index, (dataset, spec) in enumerate(protocol["datasets"].items()):
        expected_ids, manifest_provenance = _frozen_ids(spec)
        proposed, proposed_provenance = _load_per_example(
            REPO_ROOT / str(spec["proposed"]), expected_ids
        )
        baseline_scores: dict[str, dict[str, list[float]]] = {}
        baseline_provenance: dict[str, Any] = {}
        for label, relative_path in spec["baselines"].items():
            scores, provenance = _load_per_example(REPO_ROOT / str(relative_path), expected_ids)
            baseline_scores[label] = scores
            baseline_provenance[label] = provenance
        comparisons, raw_p = _paired_outcomes(
            proposed, baseline_scores, n_resamples=n_resamples,
            seed=base_seed + dataset_index * 100,
            selection_opportunities=opportunities,
        )
        all_raw_p.update({f"{dataset}:{key}": value for key, value in raw_p.items()})
        baseline_macro_means = {
            label: fmean(scores["macro_rouge"]) for label, scores in baseline_scores.items()
        }
        dataset_results[dataset] = {
            "label": spec["label"], "rows": len(expected_ids),
            "manifest": manifest_provenance, "proposed": proposed_provenance,
            "baselines": baseline_provenance,
            "adversarial_baseline": max(baseline_macro_means, key=baseline_macro_means.__getitem__),
            "comparisons": comparisons,
        }
    if len(all_raw_p) != int(protocol["multiplicity"]["holm_family_size"]):
        raise ValueError("observed Gate 2 paired family size drifted")
    adjusted = holm_adjust(all_raw_p)
    for dataset, result in dataset_results.items():
        for baseline_label, comparison in result["comparisons"].items():
            for metric, endpoint in comparison["metrics"].items():
                key = f"{dataset}:{baseline_label}:{metric}"
                endpoint["p_value_holm"] = adjusted[key]
                endpoint["descriptive_endpoint_win"] = bool(
                    endpoint["mean_difference"] > 0.0 and endpoint["ci_lower"] > 0.0
                    and endpoint["p_value_holm"] < 0.05
                )
                endpoint["selection_aware_win"] = bool(
                    endpoint["descriptive_endpoint_win"]
                    and endpoint["p_value_selection_bonferroni"] < 0.05
                )
    summary = {
        "evidence_schema_version": "1.0", "measured_at_utc": _utc_now(),
        "status": "completed_dev_diagnostic_not_promotion",
        "study_id": protocol["study_id"], "implementation_commit": _git_commit(),
        "preregistration_path": PREREGISTRATION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "scientific_config_sha256": _canonical_sha256(protocol),
        "partition": "dev", "dev_test_accessed": False, "test_split_accessed": False,
        "bootstrap": protocol["bootstrap"], "multiplicity": protocol["multiplicity"],
        "datasets": dataset_results,
        "environment": {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy")},
        "interpretation": (
            "Post-score frozen-dev diagnostic only. Holm-adjusted descriptive endpoints "
            "do not remove dev selection bias; no result authorizes dev-test or test."
        ),
    }
    _write_json(OUTPUT_ROOT / "summary.json", summary)
    _append_search_log({
        "logged_at_utc": _utc_now(), "study_id": protocol["study_id"],
        "dataset": "Multi-News+GovReport", "partition": "dev",
        "family": "gate2_paired_finalists", "candidate": "S02b_vs_8_family_finalists",
        "method": "paired_bootstrap_existing_per_example", "config_path": PREREGISTRATION,
        "config_hash": _canonical_sha256(protocol), "run_attempt": "final",
        "dev_score": None, "dev_test_score": None, "status": "completed",
        "promoted": False, "reason": "post-score dev diagnostic only; cannot authorize promotion",
        "comparison_family_size": 64, "dev_test_accessed": False, "test_split_accessed": False,
    })
    return summary


def main() -> None:
    summary = run()
    compact = {
        dataset: {"adversarial_baseline": result["adversarial_baseline"],
                  "comparison": result["comparisons"][result["adversarial_baseline"]]}
        for dataset, result in summary["datasets"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
