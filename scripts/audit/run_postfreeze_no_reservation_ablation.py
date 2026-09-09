"""Run post-freeze, development-only manuscript mechanism checks.

The script has no test-split entry point. Each registered variant changes one
candidate-generation mechanism from the frozen development anchor. Results are
supplemental evidence and cannot be used to reselect a method.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from scripts.audit.run_d3a_router_fusion_full_dev import _run_candidate, _selected_rows
from scripts.audit.run_govreport_e3_ablation import (
    METRICS,
    ProcessTreeSampler,
    _per_scores,
    _run_fixed_candidate,
    _validated_dev_partition,
    _validate_anchor_dependency_parity,
)
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
)
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.utils.io import load_yaml, read_jsonl


STUDIES = {
    "no_reservation": {
        "id": "postfreeze-no-reservation-ablation-v1",
        "preregistration": REPO_ROOT / "configs/preregistrations/postfreeze_no_reservation_ablation_v1.json",
        "output_root": REPO_ROOT / "runs_v2/postfreeze_no_reservation_v1",
        "variant": "A06_no_route_reservation",
        "comparison": "full_anchor_minus_no_route_reservation",
    },
    "no_lexical_route": {
        "id": "postfreeze-no-lexical-route-ablation-v1",
        "preregistration": REPO_ROOT / "configs/preregistrations/postfreeze_no_lexical_route_ablation_v1.json",
        "output_root": REPO_ROOT / "runs_v2/postfreeze_no_lexical_route_v1",
        "variant": "A07_no_lexical_route",
        "comparison": "full_anchor_minus_no_lexical_candidate_route",
    },
    "zero_lexical_weight_exact_pool": {
        "id": "postfreeze-zero-lexical-weight-exact-pool-v1",
        "preregistration": REPO_ROOT / "configs/preregistrations/postfreeze_zero_lexical_weight_exact_pool_v1.json",
        "output_root": REPO_ROOT / "runs_v2/postfreeze_zero_lexical_weight_v1",
        "variant": "A08_exact_pool_zero_lexical_weight",
        "comparison": "full_anchor_minus_exact_pool_zero_lexical_weight",
    },
}


def _apply_variant(config: dict[str, Any], study_key: str, dataset_key: str) -> None:
    if study_key == "no_reservation":
        config["candidate_budget"]["min_per_route"] = 0
        return
    if study_key == "no_lexical_route":
        config["compute_budget"]["enabled_routes"] = ["semantic", "graph"]
        graph_weight = 2.0 if dataset_key == "multinews" else 1.0
        config["candidates"]["route_weights"] = {
            "semantic": 1.0,
            "graph": graph_weight,
        }
        return
    if study_key == "zero_lexical_weight_exact_pool":
        graph_weight = 2.0 if dataset_key == "multinews" else 1.0
        config["candidates"]["route_weights"] = {
            "lexical": 0.0,
            "semantic": 1.0,
            "graph": graph_weight,
        }
        return
    raise ValueError(f"unknown post-freeze study: {study_key}")


def _load_fixed_pools(
    path: Path, ordered_ids: Sequence[str]
) -> dict[str, list[int]]:
    pools: dict[str, list[int]] = {}
    for row in read_jsonl(str(path)):
        row_id = str(row.get("id"))
        indices = [
            int(record["original_index"])
            for record in row.get("candidate_records") or []
        ]
        if len(indices) != len(set(indices)):
            raise ValueError(f"anchor candidate pool repeats an index for {row_id}")
        pools[row_id] = indices
    if set(pools) != set(ordered_ids):
        raise ValueError("anchor prediction pools do not match frozen development IDs")
    return pools


def _candidate_stats(path: Path) -> dict[str, float | int]:
    rows = candidates = exclusive = selected = selected_exclusive = reserved = 0
    for row in read_jsonl(str(path)):
        rows += 1
        records = list(row.get("candidate_records") or [])
        by_index = {int(record["original_index"]): record for record in records}
        candidates += len(records)
        exclusive += sum(
            int(record.get("route_agreement", 0)) == 1 for record in records
        )
        reserved += sum(
            any(
                str(reason).startswith("reserve:")
                for reason in record.get("inclusion_reasons") or []
            )
            for record in records
        )
        chosen = [int(index) for index in row.get("selected_indices") or []]
        selected += len(chosen)
        selected_exclusive += sum(
            int(by_index[index].get("route_agreement", 0)) == 1
            for index in chosen
            if index in by_index
        )
    return {
        "rows": rows,
        "mean_candidate_count": candidates / rows,
        "exclusive_candidate_fraction": exclusive / candidates if candidates else 0.0,
        "selected_exclusive_fraction": (
            selected_exclusive / selected if selected else 0.0
        ),
        "reserved_candidate_records": reserved,
    }


def _analyze(
    anchor_summary_path: Path,
    anchor_predictions_path: Path,
    variant_evidence: Mapping[str, Any],
    ordered_ids: Sequence[str],
    seed: int,
    comparison: str,
    mechanism_label: str,
) -> dict[str, Any]:
    anchor_summary = json.loads(anchor_summary_path.read_text(encoding="utf-8"))
    anchor = _per_scores(REPO_ROOT / anchor_summary["per_example_path"], ordered_ids)
    variant = _per_scores(
        REPO_ROOT / str(variant_evidence["per_example_path"]), ordered_ids
    )
    comparisons: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    for offset, metric in enumerate(METRICS):
        result = paired_bootstrap_difference(
            anchor[metric],
            variant[metric],
            n_resamples=100_000,
            seed=seed + offset,
        )
        comparisons[metric] = result
        raw_p[metric] = float(result["p_value_two_sided"])
    adjusted = holm_adjust(raw_p)
    for metric, result in comparisons.items():
        result["p_value_holm_4"] = adjusted[metric]
    return {
        "comparison": comparison,
        "comparisons": comparisons,
        "mechanism": {
            "full_anchor": _candidate_stats(anchor_predictions_path),
            mechanism_label: _candidate_stats(
                REPO_ROOT / str(variant_evidence["predictions_path"])
            ),
        },
        "interpretation_status": "post_hoc_supplemental_not_for_method_selection",
    }


def _analyze_zero_lexical_weight(
    anchor_summary_path: Path,
    anchor_predictions_path: Path,
    variant_evidence: Mapping[str, Any],
    no_lexical_spec: Mapping[str, Any],
    ordered_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    anchor_summary = json.loads(anchor_summary_path.read_text(encoding="utf-8"))
    score_sets = {
        "full_anchor": _per_scores(
            REPO_ROOT / anchor_summary["per_example_path"], ordered_ids
        ),
        "exact_pool_zero_lexical_weight": _per_scores(
            REPO_ROOT / str(variant_evidence["per_example_path"]), ordered_ids
        ),
        "no_lexical_route": _per_scores(
            REPO_ROOT / str(no_lexical_spec["per_example_path"]), ordered_ids
        ),
    }
    directions = {
        "full_anchor_minus_exact_pool_zero_lexical_weight": (
            "full_anchor", "exact_pool_zero_lexical_weight"
        ),
        "exact_pool_zero_lexical_weight_minus_no_lexical_route": (
            "exact_pool_zero_lexical_weight", "no_lexical_route"
        ),
    }
    comparisons: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    endpoint = 0
    for label, (left_name, right_name) in directions.items():
        comparisons[label] = {}
        for metric in METRICS:
            result = paired_bootstrap_difference(
                score_sets[left_name][metric],
                score_sets[right_name][metric],
                n_resamples=100_000,
                seed=seed + endpoint,
            )
            comparisons[label][metric] = result
            raw_p[f"{label}:{metric}"] = float(result["p_value_two_sided"])
            endpoint += 1
    adjusted = holm_adjust(raw_p)
    for label, values in comparisons.items():
        for metric, result in values.items():
            result["p_value_holm_8_within_dataset"] = adjusted[f"{label}:{metric}"]
    return {
        "comparisons": comparisons,
        "macro_means": {
            name: sum(scores["macro_rouge"]) / len(scores["macro_rouge"])
            for name, scores in score_sets.items()
        },
        "mechanism": {
            "full_anchor": _candidate_stats(anchor_predictions_path),
            "exact_pool_zero_lexical_weight": _candidate_stats(
                REPO_ROOT / str(variant_evidence["predictions_path"])
            ),
            "no_lexical_route": _candidate_stats(
                REPO_ROOT / str(no_lexical_spec["predictions_path"])
            ),
        },
        "interpretation_status": "post_hoc_supplemental_not_for_method_selection",
    }


def _run_dataset(
    dataset_key: str,
    specification: Mapping[str, Any],
    *,
    study_key: str,
    study_spec: Mapping[str, Any],
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    label = str(specification["label"])
    anchor_config = REPO_ROOT / specification["anchor_config"]["path"]
    anchor_predictions = REPO_ROOT / specification["anchor_predictions"]["path"]
    anchor_summary = REPO_ROOT / str(specification["anchor_summary"])
    cache_root = REPO_ROOT / str(specification["embedding_cache"])
    if sha256_file(str(anchor_config)) != specification["anchor_config"]["sha256"]:
        raise ValueError(f"{label} anchor config SHA drifted")
    if (
        sha256_file(str(anchor_predictions))
        != specification["anchor_predictions"]["sha256"]
    ):
        raise ValueError(f"{label} anchor predictions SHA drifted")
    if not anchor_summary.is_file() or not cache_root.is_dir():
        raise ValueError(f"{label} anchor summary or embedding cache is missing")
    _validate_anchor_dependency_parity(
        json.loads(anchor_summary.read_text(encoding="utf-8"))
    )

    study = _load_study(dataset_key)
    input_path = REPO_ROOT / study["input"]
    _partition, ordered_ids = _validated_dev_partition(study)
    gold = _load_gold(input_path, ordered_ids)
    config = copy.deepcopy(load_yaml(str(anchor_config)))
    _apply_variant(config, study_key, dataset_key)
    preregistration = Path(study_spec["preregistration"])
    output_root = Path(study_spec["output_root"])
    variant = str(study_spec["variant"])
    study_id = str(study_spec["id"])
    config["study"] = {
        "study_id": study_id,
        "variant": variant,
        "partition": "dev",
        "preregistration_path": _relative(preregistration),
        "postfreeze_supplemental": True,
        "method_selection_permitted": False,
    }

    root = output_root / dataset_key / "dev" / variant
    evidence_path = root / "evidence.json"
    if evidence_path.is_file():
        if not resume:
            raise ValueError(f"output already exists for {label}; pass --resume")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("status") == "completed":
            return evidence
        attempt = 1
        while (root / f"evidence.failed_attempt_{attempt:02d}.json").exists():
            attempt += 1
        failed_evidence = root / f"evidence.failed_attempt_{attempt:02d}.json"
        evidence_path.replace(failed_evidence)
        temporary = root / "predictions.jsonl.tmp"
        if temporary.exists():
            temporary.replace(root / f"predictions.failed_attempt_{attempt:02d}.jsonl.tmp")
        _append_search_log({
            "logged_at_utc": _utc_now(),
            "study_id": study_id,
            "dataset": label,
            "partition": "dev",
            "family": "postfreeze_supplemental_ablation_not_search",
            "candidate": variant,
            "status": "failed",
            "promoted": False,
            "reason": str(evidence.get("failure", "unknown preserved failure")),
            "comparison_family_size": (
                16 if study_key == "zero_lexical_weight_exact_pool" else 4
            ),
            "test_split_accessed": False,
            "preserved_evidence_path": _relative(failed_evidence),
        })

    root.mkdir(parents=True, exist_ok=resume)
    config_path = root / "resolved_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    os.environ["META_SUM_EMBEDDING_CACHE_DIR"] = str(cache_root)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    started_at = _utc_now()
    wall_started = time.perf_counter()
    try:
        with ProcessTreeSampler() as sampler:
            if study_key == "zero_lexical_weight_exact_pool":
                pools = _load_fixed_pools(anchor_predictions, ordered_ids)
                metrics, artifacts = _run_fixed_candidate(
                    rows=_selected_rows(input_path, ordered_ids),
                    ordered_ids=ordered_ids,
                    gold=gold,
                    config=config,
                    pools=pools,
                    route_weights=config["candidates"]["route_weights"],
                    candidate_root=root,
                    workers=workers,
                )
            else:
                metrics, artifacts = _run_candidate(
                    rows=_selected_rows(input_path, ordered_ids),
                    ordered_ids=ordered_ids,
                    gold=gold,
                    config=config,
                    candidate_root=root,
                    workers=workers,
                )
        evidence = {
            "evidence_schema_version": "1.0",
            "measured_at_utc": _utc_now(),
            "started_at_utc": started_at,
            "status": "completed",
            "study_id": study_id,
            "variant": variant,
            "dataset": label,
            "partition": "frozen development",
            "rows": len(ordered_ids),
            "config_path": _relative(config_path),
            "config_sha256": sha256_file(str(config_path)),
            "candidate_hash": _canonical_sha256({
                "dataset": label,
                "variant": variant,
                "config_sha256": sha256_file(str(config_path)),
                "partition_ids": selected_ids_sha256(ordered_ids),
            }),
            "implementation_commit": _git_commit(),
            "dependency_versions": _dependency_versions(),
            "execution": {
                "workers": workers,
                "threads_per_worker": 1,
                "wall_seconds": time.perf_counter() - wall_started,
                "cpu_process_tree_seconds": max(
                    0.0, sampler.cpu_end - sampler.cpu_start
                ),
                "peak_process_tree_rss_bytes": sampler.peak_rss,
                "embedding_cache_root": _relative(cache_root),
            },
            "metrics": metrics,
            **artifacts,
            "postfreeze_supplemental": True,
            "method_selection_permitted": False,
            "fixed_candidate_pool": study_key == "zero_lexical_weight_exact_pool",
            "dev_test_accessed": False,
            "test_split_accessed": False,
        }
    except Exception as error:
        _write_json(evidence_path, {
            "evidence_schema_version": "1.0",
            "measured_at_utc": _utc_now(),
            "started_at_utc": started_at,
            "status": "failed",
            "study_id": study_id,
            "variant": variant,
            "dataset": label,
            "failure": f"{type(error).__name__}: {error}",
            "dev_test_accessed": False,
            "test_split_accessed": False,
        })
        raise
    _write_json(evidence_path, evidence)
    _append_search_log({
        "logged_at_utc": _utc_now(),
        "study_id": study_id,
        "dataset": label,
        "partition": "dev",
        "family": "postfreeze_supplemental_ablation_not_search",
        "candidate": variant,
        "candidate_hash": evidence["candidate_hash"],
        "config_path": evidence["config_path"],
        "config_hash": evidence["config_sha256"],
        "dev_score": evidence["metrics"]["macro_rouge"],
        "dev_test_score": None,
        "status": "completed",
        "promoted": False,
        "reason": "post-freeze mechanism check; cannot change method selection",
        "comparison_family_size": (
            16 if study_key == "zero_lexical_weight_exact_pool" else 4
        ),
        "test_split_accessed": False,
    })
    return evidence


def run(
    datasets: Sequence[str], *, study_key: str, workers: int, resume: bool
) -> dict[str, Any]:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    study_spec = STUDIES[study_key]
    preregistration = Path(study_spec["preregistration"])
    output_root = Path(study_spec["output_root"])
    variant = str(study_spec["variant"])
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    disclosure = prereg["postfreeze_disclosure"]
    score_blind_key = (
        "zero_lexical_weight_scores_observed_before_registration"
        if study_key == "zero_lexical_weight_exact_pool"
        else "new_variant_scores_observed_before_registration"
    )
    if disclosure.get(score_blind_key) is not False:
        raise ValueError("new variant was not registered score-blind")
    if disclosure.get("method_selection_permitted") is not False:
        raise ValueError("post-freeze method selection must remain forbidden")
    if disclosure.get("test_split_accessed") is not False:
        raise ValueError("test guard is missing")

    evidence_by_dataset: dict[str, Any] = {}
    analyses: dict[str, Any] = {}
    for dataset_key in datasets:
        specification = prereg["datasets"][dataset_key]
        evidence = _run_dataset(
            dataset_key,
            specification,
            study_key=study_key,
            study_spec=study_spec,
            workers=workers,
            resume=resume,
        )
        evidence_by_dataset[dataset_key] = evidence
        study = _load_study(dataset_key)
        _partition, ordered_ids = _validated_dev_partition(study)
        if study_key == "zero_lexical_weight_exact_pool":
            no_lexical = specification["no_lexical_route"]
            if sha256_file(str(REPO_ROOT / no_lexical["predictions_path"])) != no_lexical["predictions_sha256"]:
                raise ValueError(f"{specification['label']} no-lexical predictions drifted")
            if sha256_file(str(REPO_ROOT / no_lexical["per_example_path"])) != no_lexical["per_example_sha256"]:
                raise ValueError(f"{specification['label']} no-lexical scores drifted")
            analyses[dataset_key] = _analyze_zero_lexical_weight(
                REPO_ROOT / specification["anchor_summary"],
                REPO_ROOT / specification["anchor_predictions"]["path"],
                evidence,
                no_lexical,
                ordered_ids,
                int(specification["bootstrap_seed"]),
            )
        else:
            analyses[dataset_key] = _analyze(
                REPO_ROOT / specification["anchor_summary"],
                REPO_ROOT / specification["anchor_predictions"]["path"],
                evidence,
                ordered_ids,
                int(specification["bootstrap_seed"]),
                str(study_spec["comparison"]),
                study_key,
            )

    if study_key == "zero_lexical_weight_exact_pool":
        global_raw_p = {}
        for dataset_key, analysis in analyses.items():
            for comparison, values in analysis["comparisons"].items():
                for metric, result in values.items():
                    global_raw_p[f"{dataset_key}:{comparison}:{metric}"] = float(
                        result["p_value_two_sided"]
                    )
        if len(global_raw_p) != 16:
            raise ValueError(
                "zero-lexical-weight global Holm family must contain 16 endpoints"
            )
        global_adjusted = holm_adjust(global_raw_p)
        for dataset_key, analysis in analyses.items():
            for comparison, values in analysis["comparisons"].items():
                for metric, result in values.items():
                    result["p_value_holm_16_global"] = global_adjusted[
                        f"{dataset_key}:{comparison}:{metric}"
                    ]

    output = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed",
        "study_id": str(study_spec["id"]),
        "preregistration_path": _relative(preregistration),
        "preregistration_sha256": sha256_file(str(preregistration)),
        "datasets": analyses,
        "variant_evidence": {
            key: {
                "path": _relative(
                    output_root / key / "dev" / variant / "evidence.json"
                ),
                "sha256": sha256_file(str(
                    output_root / key / "dev" / variant / "evidence.json"
                )),
            }
            for key in evidence_by_dataset
        },
        "postfreeze_supplemental": True,
        "method_selection_permitted": False,
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    _write_json(output_root / "analysis.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("govreport", "multinews", "both"), default="both"
    )
    parser.add_argument("--study", choices=tuple(STUDIES), default="no_reservation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    datasets = (
        ("govreport", "multinews")
        if args.dataset == "both"
        else (args.dataset,)
    )
    result = run(
        datasets, study_key=args.study, workers=args.workers, resume=args.resume
    )
    print(json.dumps({
        "status": result["status"],
        "datasets": list(result["datasets"]),
        "method_selection_permitted": False,
    }))


if __name__ == "__main__":
    main()
