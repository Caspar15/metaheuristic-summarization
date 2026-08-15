"""Run the preregistered GovReport E3 route/provenance ablation on dev only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import psutil
import yaml

from scripts.audit.run_d3a_router_fusion_full_dev import (
    _run_candidate,
    _selected_rows,
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
    _write_jsonl,
)
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.eval.rouge import DEFAULT_METRICS, rouge_scores
from src.pipeline.select_sentences import summarize_one
from src.utils.io import load_yaml, read_jsonl


PARENT = REPO_ROOT / "configs/preregistrations/govreport_centered_evidence_completion_v1.json"
ADDENDUM = REPO_ROOT / "configs/preregistrations/govreport_e3_execution_addendum_v1.json"
OUTPUT_ROOT = REPO_ROOT / "runs_v2/govreport_e3_route_provenance_v1/dev"
ANCHOR_CONFIG = REPO_ROOT / "runs_v2/d3b_cross_profile_combination_v1/govreport/dev/C01_combined_salience_route_weight/resolved_config.yaml"
ANCHOR_PREDICTIONS = REPO_ROOT / "runs_v2/d3b_cross_profile_combination_v1/govreport/dev/C01_combined_salience_route_weight/predictions.jsonl"
ANCHOR_SUMMARY = REPO_ROOT / "runs_v2/d3b_cross_profile_combination_v1/govreport/dev/C01_combined_salience_route_weight/candidate_summary.json"
VARIANT_ORDER = (
    "A01_no_semantic",
    "A02_no_graph",
    "A03_lexical_only_capacity_80",
    "A04_equal_rank_rrf",
    "A05_index_only_lexical",
)
METRICS = (*DEFAULT_METRICS, "macro_rouge")


class ProcessTreeSampler:
    def __init__(self, interval_seconds: float = 0.05):
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.peak_rss = 0
        self.cpu_start = 0.0
        self.cpu_end = 0.0
        self.peak_observed_cpu_total = 0.0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    @staticmethod
    def _tree_snapshot() -> tuple[int, float]:
        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        rss = 0
        cpu = 0.0
        for process in processes:
            try:
                rss += process.memory_info().rss
                times = process.cpu_times()
                cpu += times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rss, cpu

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            rss, cpu = self._tree_snapshot()
            self.peak_rss = max(self.peak_rss, rss)
            self.peak_observed_cpu_total = max(self.peak_observed_cpu_total, cpu)
            self.stop_event.wait(self.interval)

    def __enter__(self):
        rss, self.cpu_start = self._tree_snapshot()
        self.peak_rss = rss
        self.peak_observed_cpu_total = self.cpu_start
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        rss, self.cpu_end = self._tree_snapshot()
        self.peak_rss = max(self.peak_rss, rss)
        self.cpu_end = max(self.cpu_end, self.peak_observed_cpu_total)


def _validated_dev_partition(study: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return the frozen dev partition after validating its recorded identity."""
    manifest = study.get("manifest_object")
    if not isinstance(manifest, Mapping):
        raise ValueError("E3 study has no loaded partition manifest")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("E3 partition manifest has no partitions mapping")
    partition = partitions.get("dev")
    if not isinstance(partition, dict):
        raise ValueError("E3 partition manifest has no dev object")
    selected_ids = partition.get("selected_ids")
    if not isinstance(selected_ids, list) or not all(
        isinstance(value, str) for value in selected_ids
    ):
        raise ValueError("E3 dev partition has invalid selected_ids")
    ordered_ids = list(selected_ids)
    if len(ordered_ids) != partition.get("rows"):
        raise ValueError("E3 dev partition row count drifted")
    if selected_ids_sha256(ordered_ids) != partition.get("selected_ids_sha256"):
        raise ValueError("E3 dev membership drifted")
    return partition, ordered_ids


def _selected_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            {"id": row["id"], "selected_indices": row["selected_indices"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_variants(base: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    variants = {name: copy.deepcopy(dict(base)) for name in VARIANT_ORDER}
    variants["A01_no_semantic"]["compute_budget"]["enabled_routes"] = ["lexical", "graph"]
    variants["A01_no_semantic"]["candidates"]["route_weights"] = {"lexical": 0.5, "graph": 1.0}
    variants["A02_no_graph"]["compute_budget"]["enabled_routes"] = ["lexical", "semantic"]
    variants["A02_no_graph"]["candidates"]["route_weights"] = {"lexical": 0.5, "semantic": 1.0}
    variants["A03_lexical_only_capacity_80"]["compute_budget"]["enabled_routes"] = ["lexical"]
    variants["A03_lexical_only_capacity_80"]["candidate_budget"] = {
        "route_top_k": 80, "min_per_route": 0, "total": 80
    }
    variants["A03_lexical_only_capacity_80"]["candidates"]["route_weights"] = {"lexical": 1.0}
    variants["A04_equal_rank_rrf"]["candidates"]["route_weights"] = {
        "lexical": 1.0, "semantic": 1.0, "graph": 1.0
    }
    variants["A05_index_only_lexical"]["selector"]["salience_source"] = "lexical_percentile"
    for name, config in variants.items():
        config["study"] = {
            "study_id": "govreport-e3-route-provenance-v1",
            "variant": name,
            "partition": "dev",
            "preregistration_path": _relative(PARENT),
            "execution_addendum_path": _relative(ADDENDUM),
        }
    return variants


def _load_anchor_pools(ordered_ids: Sequence[str]) -> dict[str, list[int]]:
    pools: dict[str, list[int]] = {}
    for row in read_jsonl(str(ANCHOR_PREDICTIONS)):
        row_id = str(row.get("id"))
        records = list(row.get("candidate_records") or [])
        indices = [int(record["original_index"]) for record in records]
        if len(indices) != len(set(indices)):
            raise ValueError(f"anchor candidate pool repeats an index for {row_id}")
        pools[row_id] = indices
    if set(pools) != set(ordered_ids):
        raise ValueError("anchor prediction pools do not match frozen GovReport dev")
    return pools


def _fixed_task(payload):
    doc, config, fixed_indices, route_weights = payload
    return summarize_one(
        doc,
        config,
        fixed_candidate_original_indices=fixed_indices,
        audit_route_weights=route_weights,
    )


def _bounded_fixed_rows(
    rows: Iterator[dict[str, Any]],
    config: Mapping[str, Any],
    pools: Mapping[str, Sequence[int]],
    route_weights: Mapping[str, float] | None,
    *,
    workers: int,
) -> Iterator[dict[str, Any]]:
    if workers == 1:
        for row in rows:
            yield _fixed_task((row, config, list(pools[row["id"]]), route_weights))
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: dict[Any, int] = {}
        ready: dict[int, dict[str, Any]] = {}
        submitted = 0
        yielded = 0
        exhausted = False
        while not exhausted or pending:
            while not exhausted and len(pending) < workers * 2:
                try:
                    row = next(rows)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(
                    _fixed_task,
                    (row, config, list(pools[row["id"]]), route_weights),
                )
                pending[future] = submitted
                submitted += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                ready[pending.pop(future)] = future.result()
            while yielded in ready:
                yield ready.pop(yielded)
                yielded += 1


def _run_fixed_candidate(
    *,
    rows: Iterator[dict[str, Any]],
    ordered_ids: Sequence[str],
    gold: Mapping[str, Sequence[str]],
    config: dict[str, Any],
    pools: Mapping[str, Sequence[int]],
    route_weights: Mapping[str, float] | None,
    candidate_root: Path,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions_path = candidate_root / "predictions.jsonl"
    temporary = predictions_path.with_suffix(".jsonl.tmp")
    summaries: list[str] = []
    feasibility: list[bool] = []
    lengths: list[int] = []
    selected_rows: list[dict[str, Any]] = []
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        iterator = _bounded_fixed_rows(
            rows, config, pools, route_weights, workers=workers
        )
        for position, prediction in enumerate(iterator):
            if prediction.get("id") != ordered_ids[position]:
                raise RuntimeError("E3 fixed-pool prediction order drifted")
            actual_pool = [
                int(record["original_index"])
                for record in prediction.get("candidate_records") or []
            ]
            if actual_pool != sorted(pools[prediction["id"]]):
                raise RuntimeError("E3 fixed candidate membership drifted")
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            summaries.append(str(prediction.get("summary", "")))
            feasibility.append(bool(prediction.get("feasible")))
            lengths.append(len(str(prediction.get("summary", "")).split()))
            selected_rows.append({
                "id": prediction["id"],
                "selected_indices": prediction["selected_indices"],
            })
        handle.flush()
        os.fsync(handle.fileno())
    if len(summaries) != len(ordered_ids):
        raise RuntimeError("E3 fixed-pool prediction count drifted")
    os.replace(temporary, predictions_path)
    means, per_example = rouge_scores(
        summaries,
        [gold[row_id] for row_id in ordered_ids],
        metrics=DEFAULT_METRICS,
        return_per_example=True,
    )
    per_rows = [
        {"id": row_id, **scores}
        for row_id, scores in zip(ordered_ids, per_example)
    ]
    per_path = candidate_root / "per_example.jsonl"
    _write_jsonl(per_path, per_rows)
    metrics = {
        "evaluation_protocol": "multisentence_lsum",
        "rows": len(ordered_ids),
        "rouge": means,
        "macro_rouge": fmean(float(means[name]) for name in DEFAULT_METRICS),
        "feasible_rows": int(sum(feasibility)),
        "infeasible_rows": int(len(feasibility) - sum(feasibility)),
        "summary_words": {
            "mean": float(np.mean(lengths)),
            "min": int(min(lengths)),
            "max": int(max(lengths)),
        },
    }
    _write_json(candidate_root / "metrics.json", metrics)
    return metrics, {
        "predictions_path": _relative(predictions_path),
        "predictions_sha256": sha256_file(str(predictions_path)),
        "selected_indices_sha256": _selected_digest(selected_rows),
        "per_example_path": _relative(per_path),
        "per_example_sha256": sha256_file(str(per_path)),
    }


def _per_scores(path: Path, ordered_ids: Sequence[str]) -> dict[str, list[float]]:
    rows = list(read_jsonl(str(path)))
    if [row.get("id") for row in rows] != list(ordered_ids):
        raise ValueError(f"E3 per-example order mismatch: {_relative(path)}")
    result = {metric: [] for metric in METRICS}
    for row in rows:
        components = [float(row[name]) for name in DEFAULT_METRICS]
        for name, value in zip(DEFAULT_METRICS, components):
            result[name].append(value)
        result["macro_rouge"].append(fmean(components))
    return result


def _analyze(results: Mapping[str, Mapping[str, Any]], ordered_ids: Sequence[str]) -> dict:
    anchor_summary = json.loads(ANCHOR_SUMMARY.read_text(encoding="utf-8"))
    anchor = _per_scores(REPO_ROOT / anchor_summary["per_example_path"], ordered_ids)
    raw_p: dict[str, float] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    endpoint = 0
    for name in VARIANT_ORDER:
        scores = _per_scores(REPO_ROOT / results[name]["per_example_path"], ordered_ids)
        comparisons[name] = {}
        for metric in METRICS:
            value = paired_bootstrap_difference(
                anchor[metric],
                scores[metric],
                n_resamples=100_000,
                seed=20260820 + endpoint,
            )
            key = f"C01-{name}:{metric}"
            raw_p[key] = float(value["p_value_two_sided"])
            comparisons[name][metric] = value
            endpoint += 1
    if len(raw_p) != 20:
        raise ValueError("E3 Holm family must contain exactly 20 endpoints")
    adjusted = holm_adjust(raw_p)
    for name, endpoints in comparisons.items():
        for metric, value in endpoints.items():
            value["p_value_holm_20"] = adjusted[f"C01-{name}:{metric}"]
            value["significant_positive_after_holm"] = bool(
                float(value["ci_lower"]) > 0
                and float(value["p_value_holm_20"]) <= 0.05
            )
    decisions = {
        "semantic_quality_contribution": comparisons["A01_no_semantic"]["macro_rouge"]["significant_positive_after_holm"],
        "graph_quality_contribution": comparisons["A02_no_graph"]["macro_rouge"]["significant_positive_after_holm"],
        "combined_nonlexical_contribution": comparisons["A03_lexical_only_capacity_80"]["macro_rouge"]["significant_positive_after_holm"],
        "provenance_equal_rrf_contribution": comparisons["A04_equal_rank_rrf"]["macro_rouge"]["significant_positive_after_holm"],
        "provenance_vs_lexical_only_contribution": comparisons["A05_index_only_lexical"]["macro_rouge"]["significant_positive_after_holm"],
    }
    return {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed",
        "study_id": "govreport-e3-route-provenance-v1",
        "partition": "GovReport frozen dev",
        "rows": len(ordered_ids),
        "holm_family_size": 20,
        "bootstrap_resamples": 100_000,
        "base_seed": 20260820,
        "anchor_metrics": anchor_summary["metrics"],
        "comparisons_full_minus_ablation": comparisons,
        "claim_decisions": decisions,
        "required_claim_downgrades": [key for key, value in decisions.items() if not value],
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }


def run(*, workers: int, resume: bool = False) -> dict[str, Any]:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    if parent.get("test_split_accessed") is not False or addendum.get("test_split_accessed") is not False:
        raise ValueError("E3 protected-test guard is missing")
    if addendum.get("scores_observed_before_registration") is not False:
        raise ValueError("E3 execution addendum is not score-blind")
    anchor = addendum["anchor"]
    if sha256_file(str(ANCHOR_CONFIG)) != anchor["config_sha256"]:
        raise ValueError("E3 anchor config SHA drifted")
    if sha256_file(str(ANCHOR_PREDICTIONS)) != anchor["prediction_sha256"]:
        raise ValueError("E3 anchor predictions SHA drifted")

    study = _load_study("govreport")
    input_path = REPO_ROOT / study["input"]
    _partition, ordered_ids = _validated_dev_partition(study)
    gold = _load_gold(input_path, ordered_ids)
    base = load_yaml(str(ANCHOR_CONFIG))
    variants = _resolve_variants(base)
    pools = _load_anchor_pools(ordered_ids)
    cache_root = REPO_ROOT / "runs_v2/gate2_baseline_matrix_v1/govreport/dev/plm/_embedding_cache"
    if not cache_root.is_dir():
        raise ValueError("E3 embedding cache is missing")
    os.environ["META_SUM_EMBEDDING_CACHE_DIR"] = str(cache_root)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for name in VARIANT_ORDER:
        candidate_root = OUTPUT_ROOT / name
        evidence_path = candidate_root / "evidence.json"
        if evidence_path.is_file():
            if not resume:
                raise ValueError(f"E3 output already exists for {name}; pass --resume")
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            if evidence.get("status") != "completed":
                raise ValueError(f"failed E3 evidence is preserved for {name}")
            results[name] = evidence
            continue
        candidate_root.mkdir(parents=True, exist_ok=False)
        config = variants[name]
        config_path = candidate_root / "resolved_config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )
        started_at = _utc_now()
        wall_started = time.perf_counter()
        try:
            with ProcessTreeSampler() as sampler:
                if name in {"A04_equal_rank_rrf", "A05_index_only_lexical"}:
                    route_weights = (
                        {"lexical": 1.0, "semantic": 1.0, "graph": 1.0}
                        if name == "A04_equal_rank_rrf"
                        else None
                    )
                    metrics, artifacts = _run_fixed_candidate(
                        rows=_selected_rows(input_path, ordered_ids),
                        ordered_ids=ordered_ids,
                        gold=gold,
                        config=config,
                        pools=pools,
                        route_weights=route_weights,
                        candidate_root=candidate_root,
                        workers=workers,
                    )
                else:
                    metrics, artifacts = _run_candidate(
                        rows=_selected_rows(input_path, ordered_ids),
                        ordered_ids=ordered_ids,
                        gold=gold,
                        config=config,
                        candidate_root=candidate_root,
                        workers=workers,
                    )
            evidence = {
                "evidence_schema_version": "1.0",
                "measured_at_utc": _utc_now(),
                "started_at_utc": started_at,
                "status": "completed",
                "study_id": "govreport-e3-route-provenance-v1",
                "variant": name,
                "partition": "GovReport frozen dev",
                "rows": len(ordered_ids),
                "config_path": _relative(config_path),
                "config_sha256": sha256_file(str(config_path)),
                "candidate_hash": _canonical_sha256({
                    "variant": name,
                    "config_sha256": sha256_file(str(config_path)),
                    "partition_ids": selected_ids_sha256(ordered_ids),
                    "anchor_pool_sha256": anchor["prediction_sha256"] if name.startswith("A0") else None,
                }),
                "implementation_commit": _git_commit(),
                "dependency_versions": _dependency_versions(),
                "execution": {
                    "workers": workers,
                    "threads_per_worker": 1,
                    "wall_seconds": time.perf_counter() - wall_started,
                    "cpu_process_tree_seconds": max(0.0, sampler.cpu_end - sampler.cpu_start),
                    "peak_process_tree_rss_bytes": sampler.peak_rss,
                    "rss_sample_interval_ms": 50,
                    "embedding_cache_root": _relative(cache_root),
                },
                "metrics": metrics,
                **artifacts,
                "dev_test_accessed": False,
                "test_split_accessed": False,
            }
        except Exception as error:
            evidence = {
                "evidence_schema_version": "1.0",
                "measured_at_utc": _utc_now(),
                "started_at_utc": started_at,
                "status": "failed",
                "study_id": "govreport-e3-route-provenance-v1",
                "variant": name,
                "failure": f"{type(error).__name__}: {error}",
                "dev_test_accessed": False,
                "test_split_accessed": False,
            }
            _write_json(evidence_path, evidence)
            raise
        _write_json(evidence_path, evidence)
        _append_search_log({
            "logged_at_utc": _utc_now(),
            "study_id": "govreport-e3-route-provenance-v1",
            "dataset": "GovReport",
            "partition": "dev",
            "family": "confirmatory_ablation_not_search",
            "candidate": name,
            "candidate_hash": evidence["candidate_hash"],
            "config_path": evidence["config_path"],
            "config_hash": evidence["config_sha256"],
            "dev_score": evidence["metrics"]["macro_rouge"],
            "dev_test_score": None,
            "status": "completed",
            "promoted": None,
            "reason": "preregistered E3 evidence completion; cannot change method selection",
            "comparison_family_size": 20,
            "test_split_accessed": False,
        })
        results[name] = evidence
        print(json.dumps({"variant": name, "macro": evidence["metrics"]["macro_rouge"]}))
    analysis = _analyze(results, ordered_ids)
    _write_json(OUTPUT_ROOT.parent / "analysis.json", analysis)
    _write_json(
        OUTPUT_ROOT.parent / "study_summary.json",
        {
            "evidence_schema_version": "1.0",
            "measured_at_utc": _utc_now(),
            "status": "completed",
            "study_id": "govreport-e3-route-provenance-v1",
            "partition": "GovReport frozen dev",
            "rows": len(ordered_ids),
            "variants": results,
            "analysis_path": _relative(OUTPUT_ROOT.parent / "analysis.json"),
            "analysis_sha256": sha256_file(str(OUTPUT_ROOT.parent / "analysis.json")),
            "dev_test_accessed": False,
            "test_split_accessed": False,
        },
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(workers=args.workers, resume=args.resume)
    print(json.dumps({"status": result["status"], "downgrades": result["required_claim_downgrades"]}))


if __name__ == "__main__":
    main()
