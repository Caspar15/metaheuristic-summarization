"""Execute the preregistered A1 length-contract study on dev/dev-test only.

The script materializes an exact resolved config for every protocol, runs the
same Lead, Random, and lexical-only Greedy families, evaluates every selected
row, writes one evidence file per method run, and appends every success or
failure to ``runs_v2/search_log.jsonl``.  It refuses a second completed
dev-test observation for the same logical candidate.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file, verify_pin
from src.data.schemas import extract_references
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.eval.rouge import DEFAULT_METRICS, rouge_scores
from src.utils.io import load_yaml, read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_LOG = REPO_ROOT / "runs_v2" / "search_log.jsonl"
METHODS = ("lead", "random", "greedy")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260808

STUDIES: dict[str, dict[str, Any]] = {
    "multinews": {
        "dataset_label": "Multi-News",
        "input": "data/processed/multi_news_validation_canonical.jsonl",
        "base_config": "configs/studies/a1/multinews_base.yaml",
        "preregistration": "configs/preregistrations/a1_length_contract_v1.json",
        "manifest": "configs/validation_partitions/multinews_validation_dev_v1.json",
        "manifest_sha256": "e61405482cda203c0bd50dda3e958986b11a51b48124129e68617f97ce9e42ee",
        "candidate_key": "multi_news_candidates",
        "tie_candidate": "max_only_250",
    },
    "govreport": {
        "dataset_label": "GovReport",
        "input": "data/processed/govreport_validation_canonical.jsonl",
        "base_config": "configs/studies/a1/govreport_base.yaml",
        "preregistration": "configs/preregistrations/a1_length_contract_govreport_v1.json",
        "manifest": "configs/validation_partitions/govreport_validation_dev_v1.json",
        "manifest_sha256": "7a15ffbb87abe690fe4e72a1e0daf27bf34b3a3293371983ae8e362d06e2717e",
        "candidate_key": "candidates",
        "tie_candidate": "median_cap_570",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _append_search_log(entry: Mapping[str, Any]) -> None:
    SEARCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SEARCH_LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(entry), ensure_ascii=False) + "\n")


def _load_search_log() -> list[dict[str, Any]]:
    if not SEARCH_LOG.exists():
        return []
    return [dict(row) for row in read_jsonl(str(SEARCH_LOG))]


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for distribution in (
        "numpy",
        "scikit-learn",
        "nltk",
        "rouge-score",
        "PyYAML",
        "torch",
        "transformers",
        "tokenizers",
        "sentence-transformers",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _load_study(dataset: str) -> dict[str, Any]:
    spec = copy.deepcopy(STUDIES[dataset])
    prereg_path = REPO_ROOT / spec["preregistration"]
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("status") != "frozen_before_candidate_system_scores":
        raise ValueError("A1 preregistration is not frozen before candidate scores")
    if not prereg.get("test_split_prohibited"):
        raise ValueError("A1 preregistration must explicitly prohibit test")
    candidates = prereg.get(spec["candidate_key"])
    if not isinstance(candidates, dict) or len(candidates) != 4:
        raise ValueError("A1 preregistration must declare exactly four candidates")
    normalized_candidates: dict[str, dict[str, int]] = {}
    for name, candidate in candidates.items():
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate {name!r} must be an object")
        min_words = int(candidate["min_words"])
        max_words = int(candidate["max_words"])
        if min_words < 0 or max_words < 1 or min_words > max_words:
            raise ValueError(f"invalid length contract for {name!r}")
        normalized_candidates[name] = {
            "min_words": min_words,
            "max_words": max_words,
        }
    manifest_path = REPO_ROOT / spec["manifest"]
    pin_status = verify_pin(str(manifest_path), spec["manifest_sha256"])
    if pin_status == "legacy":
        print(f"[legacy pin] {manifest_path} (CRLF-era pin, see errata)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec.update(
        {
            "prereg": prereg,
            "preregistration_sha256": sha256_file(str(prereg_path)),
            "candidates": normalized_candidates,
            "manifest_object": manifest,
        }
    )
    return spec


def _logical_candidate_hash(
    base_config: Mapping[str, Any],
    *,
    dataset: str,
    candidate_name: str,
    length_contract: Mapping[str, int],
) -> str:
    """Hash the logical candidate without its dev/dev-test membership."""

    return _canonical_sha256(
        {
            "study": "a1-length-contract-v1",
            "dataset": dataset,
            "candidate": candidate_name,
            "base_config": base_config,
            "length_control": dict(length_contract),
            "methods": METHODS,
            "evaluation_protocol": "multisentence_lsum",
            "primary_score": "mean across methods and per-row R1/R2/RLsum",
        }
    )


def _resolved_config(
    base_config: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    dataset: str,
    partition: str,
    candidate_name: str,
    length_contract: Mapping[str, int],
    candidate_hash: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base_config))
    cfg["experiment_partition"] = {
        "manifest_path": spec["manifest"],
        "manifest_sha256": spec["manifest_sha256"],
        "name": partition,
    }
    cfg.setdefault("length_control", {})
    cfg["length_control"].update(
        {
            "unit": "words",
            "min_words": int(length_contract["min_words"]),
            "max_words": int(length_contract["max_words"]),
            "require_nonempty": True,
        }
    )
    cfg["study"] = {
        "study_id": spec["prereg"]["study_id"],
        "dataset_key": dataset,
        "candidate": candidate_name,
        "candidate_hash": candidate_hash,
        "partition": partition,
        "preregistration_path": spec["preregistration"],
        "preregistration_sha256": spec["preregistration_sha256"],
    }
    return cfg


def _selected_indices_digest(prediction_rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in prediction_rows:
        record = {
            "id": row.get("id"),
            "selected_indices": row.get("selected_indices", []),
        }
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _candidate_macro_rows(
    per_method: Mapping[str, Sequence[Mapping[str, float]]]
) -> np.ndarray:
    if set(per_method) != set(METHODS):
        raise ValueError("candidate macro requires Lead, Random, and Greedy")
    row_counts = {len(rows) for rows in per_method.values()}
    if len(row_counts) != 1:
        raise ValueError("method score rows are not aligned")
    result: list[float] = []
    for row_index in range(next(iter(row_counts))):
        values = [
            float(per_method[method][row_index][metric])
            for method in METHODS
            for metric in DEFAULT_METRICS
        ]
        result.append(float(np.mean(values)))
    return np.asarray(result, dtype=float)


def _select_protocol(
    candidate_rows: Mapping[str, Sequence[float]],
    *,
    tie_candidate: str,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    means = {
        name: float(np.asarray(values, dtype=float).mean())
        for name, values in candidate_rows.items()
    }
    raw_winner = max(means, key=lambda name: (means[name], name))
    comparisons: dict[str, dict[str, Any]] = {}
    raw_p: dict[str, float] = {}
    for comparison_index, other in enumerate(sorted(candidate_rows)):
        if other == raw_winner:
            continue
        result = paired_bootstrap_difference(
            candidate_rows[raw_winner],
            candidate_rows[other],
            n_resamples=n_resamples,
            seed=seed + comparison_index,
        )
        comparisons[other] = result
        raw_p[other] = float(result["p_value_two_sided"])
    adjusted = holm_adjust(raw_p)
    significant = True
    for other, result in comparisons.items():
        result["p_value_holm"] = adjusted[other]
        result["significant_positive_after_holm"] = bool(
            float(result["ci_lower"]) > 0.0 and adjusted[other] < 0.05
        )
        significant &= result["significant_positive_after_holm"]
    chosen = raw_winner if significant else tie_candidate
    return {
        "candidate_means": means,
        "raw_winner": raw_winner,
        "comparisons_to_raw_winner": comparisons,
        "holm_family_size": len(comparisons),
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": seed,
        "raw_winner_significant_against_all": significant,
        "selected_protocol": chosen,
        "selection_reason": (
            "raw winner passed every positive paired-bootstrap and Holm test"
            if significant
            else "no multiplicity-corrected winner; apply preregistered tie rule"
        ),
    }


def _load_gold(
    input_path: Path, selected_ids: Sequence[str]
) -> dict[str, list[str]]:
    selected = set(selected_ids)
    gold: dict[str, list[str]] = {}
    for row in read_jsonl(str(input_path)):
        row_id = row.get("id")
        if row_id in selected:
            if row_id in gold:
                raise ValueError(f"duplicate selected gold ID {row_id!r}")
            gold[row_id] = extract_references(row)
    if set(gold) != selected:
        missing = sorted(selected - set(gold))
        raise ValueError(f"gold input is missing selected IDs: {missing[:5]}")
    return gold


def _embedding_cache_summary(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Summarize per-row cache provenance without storing every cache key."""

    statuses: dict[str, int] = {}
    contract_versions: set[str] = set()
    keyed_rows: list[dict[str, str]] = []
    for row in prediction_rows:
        diagnostics = row.get("baseline_diagnostics") or {}
        representations = [
            (diagnostics.get("representation") or {}),
            ((row.get("selector_inputs") or {}).get("representation") or {}),
            (
                (row.get("optimizer_diagnostics") or {}).get(
                    "selector_representation"
                )
                or {}
            ),
        ]
        caches = [
            representation.get("embedding_cache")
            for representation in representations
            if representation.get("embedding_cache") is not None
        ]
        for record in row.get("candidate_records") or []:
            semantic = (record.get("route_scores") or {}).get("semantic") or {}
            route_cache = (semantic.get("metadata") or {}).get("embedding_cache")
            if route_cache is not None:
                caches.append(route_cache)
        cache = caches[0] if caches else None
        if any(candidate != cache for candidate in caches[1:]):
            raise ValueError("embedding cache provenance disagrees across row views")
        if cache is None:
            continue
        if not isinstance(cache, Mapping):
            raise ValueError("embedding cache provenance must be a mapping")
        status = cache.get("status")
        key = cache.get("cache_key")
        contract = cache.get("contract_version")
        if status not in {"hit", "miss_written"}:
            raise ValueError(f"unknown embedding cache status: {status!r}")
        if not isinstance(key, str) or len(key) != 64:
            raise ValueError("embedding cache provenance has no SHA-256 key")
        if not isinstance(contract, str) or not contract:
            raise ValueError("embedding cache provenance has no contract version")
        statuses[status] = statuses.get(status, 0) + 1
        contract_versions.add(contract)
        keyed_rows.append({"id": str(row.get("id")), "cache_key": key})
    if not keyed_rows:
        return None
    if len(keyed_rows) != len(prediction_rows):
        raise ValueError("embedding cache provenance is present for only some rows")
    if len(contract_versions) != 1:
        raise ValueError("embedding cache contract changed within one run")
    return {
        "rows": len(keyed_rows),
        "status_counts": statuses,
        "contract_version": next(iter(contract_versions)),
        "ordered_row_cache_keys_sha256": _canonical_sha256(keyed_rows),
    }


def _command_for_method(
    method: str,
    *,
    config_path: Path,
    input_path: Path,
    method_root: Path,
    pipeline_selector: bool = False,
) -> list[str]:
    common = [
        "--config",
        str(config_path),
        "--split",
        "validation",
        "--input",
        str(input_path),
        "--run_dir",
        str(method_root),
        "--stamp",
        "run",
    ]
    if method == "greedy" or pipeline_selector:
        return [sys.executable, "-m", "src.pipeline.select_sentences", *common]
    command = [
        sys.executable,
        "-m",
        "src.baselines.cli",
        "--baseline",
        method,
        *common,
    ]
    if method == "lead":
        command.extend(["--ordering", "document_order"])
    elif method == "random":
        command.extend(["--seed", "3407"])
    return command


def _run_method(
    method: str,
    *,
    config_path: Path,
    config_sha256: str,
    input_path: Path,
    candidate_root: Path,
    ordered_ids: Sequence[str],
    gold: Mapping[str, Sequence[str]],
    study_context: Mapping[str, Any],
    subprocess_env: Mapping[str, str] | None = None,
    pipeline_selector: bool = False,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    method_root = candidate_root / method
    run_path = method_root / "run"
    if run_path.exists():
        raise ValueError(f"refusing to overwrite existing run directory: {run_path}")
    command = _command_for_method(
        method,
        config_path=config_path,
        input_path=input_path,
        method_root=method_root,
        pipeline_selector=pipeline_selector,
    )
    started_at = _utc_now()
    execution_env = None
    if subprocess_env:
        execution_env = os.environ.copy()
        execution_env.update(
            {str(key): str(value) for key, value in subprocess_env.items()}
        )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=execution_env,
    )
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "command.log").write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
        newline="\n",
    )
    if completed.returncode != 0:
        evidence = {
            "evidence_schema_version": "1.0",
            "measured_at_utc": _utc_now(),
            "started_at_utc": started_at,
            "status": "failed",
            "test_split_accessed": False,
            "method": method,
            "command": command,
            "return_code": completed.returncode,
            "config_path": _relative(config_path),
            "config_sha256": config_sha256,
            **dict(study_context),
        }
        _write_json(run_path / "evidence.json", evidence)
        raise RuntimeError(
            f"{method} failed with exit code {completed.returncode}; "
            f"see {_relative(run_path / 'command.log')}"
        )

    predictions_path = run_path / "predictions.jsonl"
    prediction_rows = list(read_jsonl(str(predictions_path)))
    embedding_cache = _embedding_cache_summary(prediction_rows)
    expected_cache = (
        (study_context.get("execution_optimization") or {})
        .get("embedding_cache", {})
        .get("enabled", False)
    )
    if expected_cache and embedding_cache is None:
        raise ValueError(
            "PLM execution declared an embedding cache but predictions have no "
            "per-row cache provenance"
        )
    prediction_by_id: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{method} prediction has no ID")
        if row_id in prediction_by_id:
            raise ValueError(f"{method} prediction repeats {row_id!r}")
        prediction_by_id[row_id] = row
    if set(prediction_by_id) != set(ordered_ids):
        raise ValueError(f"{method} predictions do not match frozen partition IDs")
    predictions = [str(prediction_by_id[row_id].get("summary", "")) for row_id in ordered_ids]
    references = [gold[row_id] for row_id in ordered_ids]
    means, per_example = rouge_scores(
        predictions, references, return_per_example=True
    )
    per_example_rows = [
        {"id": row_id, **scores}
        for row_id, scores in zip(ordered_ids, per_example)
    ]
    _write_jsonl(run_path / "per_example.jsonl", per_example_rows)
    feasibility = [bool(prediction_by_id[row_id].get("feasible")) for row_id in ordered_ids]
    lengths = [len(prediction_by_id[row_id].get("summary", "").split()) for row_id in ordered_ids]
    metrics = {
        "evaluation_protocol": "multisentence_lsum",
        "rows": len(ordered_ids),
        "rouge": means,
        "macro_rouge": float(np.mean([means[metric] for metric in DEFAULT_METRICS])),
        "feasible_rows": int(sum(feasibility)),
        "infeasible_rows": int(len(feasibility) - sum(feasibility)),
        "summary_words": {
            "mean": float(np.mean(lengths)),
            "min": int(min(lengths)),
            "max": int(max(lengths)),
        },
    }
    _write_json(run_path / "metrics.json", metrics)
    dataset_preflight_path = run_path / "dataset_preflight.json"
    partition_preflight_path = run_path / "partition_preflight.json"
    evidence = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "started_at_utc": started_at,
        "status": "completed",
        "test_split_accessed": False,
        "method": method,
        "command": command,
        "implementation_commit": _git_commit(),
        "config_path": _relative(config_path),
        "config_sha256": config_sha256,
        "input_path": _relative(input_path),
        "input_sha256": sha256_file(str(input_path)),
        "dataset_preflight_sha256": sha256_file(str(dataset_preflight_path)),
        "partition_preflight_sha256": sha256_file(str(partition_preflight_path)),
        "predictions_path": _relative(predictions_path),
        "predictions_sha256": sha256_file(str(predictions_path)),
        "selected_indices_sha256": _selected_indices_digest(prediction_rows),
        "selected_indices_digest_semantics": "SHA-256 over canonical LF-separated {id, selected_indices}; use this, not predictions artifact SHA, for cross-machine selection reproducibility",
        "dependency_versions": _dependency_versions(),
        "metrics": metrics,
        **dict(study_context),
    }
    if embedding_cache is not None:
        evidence["embedding_cache_summary"] = embedding_cache
    _write_json(run_path / "evidence.json", evidence)
    return metrics, [dict(scores) for scores in per_example]


def run_study(dataset: str, partition: str) -> dict[str, Any]:
    if partition not in {"dev", "dev-test"}:
        raise ValueError("partition must be dev or dev-test")
    spec = _load_study(dataset)
    input_path = REPO_ROOT / spec["input"]
    base_config_path = REPO_ROOT / spec["base_config"]
    base_config = load_yaml(str(base_config_path))
    partition_entry = spec["manifest_object"]["partitions"][partition]
    ordered_ids = partition_entry["selected_ids"]
    if selected_ids_sha256(ordered_ids) != partition_entry["selected_ids_sha256"]:
        raise ValueError("frozen selected-ID digest is inconsistent")
    gold = _load_gold(input_path, ordered_ids)
    output_root = REPO_ROOT / "runs_v2" / "a1_length_contract" / dataset / partition
    if output_root.exists():
        raise ValueError(f"refusing to overwrite existing study directory: {output_root}")

    dev_summary_path = (
        REPO_ROOT
        / "runs_v2"
        / "a1_length_contract"
        / dataset
        / "dev"
        / "study_summary.json"
    )
    dev_summary = None
    if partition == "dev-test":
        if not dev_summary_path.is_file():
            raise ValueError("dev-test is forbidden before the complete dev study exists")
        dev_summary = json.loads(dev_summary_path.read_text(encoding="utf-8"))

    existing_log = _load_search_log()
    candidate_results: dict[str, dict[str, Any]] = {}
    candidate_score_rows: dict[str, np.ndarray] = {}
    for candidate_name, length_contract in spec["candidates"].items():
        candidate_hash = _logical_candidate_hash(
            base_config,
            dataset=dataset,
            candidate_name=candidate_name,
            length_contract=length_contract,
        )
        if partition == "dev-test" and any(
            row.get("study_id") == spec["prereg"]["study_id"]
            and row.get("dataset") == spec["dataset_label"]
            and row.get("candidate_hash") == candidate_hash
            and row.get("partition") == "dev-test"
            and row.get("status") == "completed"
            for row in existing_log
        ):
            raise ValueError(
                f"dev-test candidate {candidate_name!r} was already observed; refusing a second look"
            )
        candidate_root = output_root / candidate_name
        candidate_root.mkdir(parents=True, exist_ok=False)
        resolved = _resolved_config(
            base_config,
            spec=spec,
            dataset=dataset,
            partition=partition,
            candidate_name=candidate_name,
            length_contract=length_contract,
            candidate_hash=candidate_hash,
        )
        config_path = candidate_root / "resolved_config.yaml"
        config_path.write_text(
            yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )
        config_sha256 = sha256_file(str(config_path))
        context = {
            "study_id": spec["prereg"]["study_id"],
            "dataset": spec["dataset_label"],
            "partition": partition,
            "partition_rows": len(ordered_ids),
            "partition_manifest_path": spec["manifest"],
            "partition_manifest_sha256": spec["manifest_sha256"],
            "partition_selected_ids_sha256": partition_entry["selected_ids_sha256"],
            "preregistration_path": spec["preregistration"],
            "preregistration_sha256": spec["preregistration_sha256"],
            "candidate": candidate_name,
            "candidate_hash": candidate_hash,
            "length_contract": length_contract,
        }
        per_method: dict[str, list[dict[str, float]]] = {}
        method_metrics: dict[str, dict[str, Any]] = {}
        failure: str | None = None
        for method in METHODS:
            try:
                metrics, per_example = _run_method(
                    method,
                    config_path=config_path,
                    config_sha256=config_sha256,
                    input_path=input_path,
                    candidate_root=candidate_root,
                    ordered_ids=ordered_ids,
                    gold=gold,
                    study_context=context,
                )
                method_metrics[method] = metrics
                per_method[method] = per_example
            except Exception as error:  # log the failed config, then continue study
                failure = f"{type(error).__name__}: {error}"
                break

        if failure is None:
            macro_rows = _candidate_macro_rows(per_method)
            candidate_score_rows[candidate_name] = macro_rows
            per_example_path = candidate_root / "per_example.jsonl"
            _write_jsonl(
                per_example_path,
                [
                    {"id": row_id, "cross_method_macro_rouge": float(score)}
                    for row_id, score in zip(ordered_ids, macro_rows)
                ],
            )
            result = {
                "status": "completed",
                **context,
                "resolved_config_path": _relative(config_path),
                "resolved_config_sha256": config_sha256,
                "methods": method_metrics,
                "cross_method_macro_rouge": float(macro_rows.mean()),
                "per_example_sha256": sha256_file(str(per_example_path)),
            }
        else:
            result = {
                "status": "failed",
                **context,
                "resolved_config_path": _relative(config_path),
                "resolved_config_sha256": config_sha256,
                "methods_completed": sorted(method_metrics),
                "failure": failure,
            }
        _write_json(candidate_root / "candidate_summary.json", result)
        candidate_results[candidate_name] = result

    completed_candidates = {
        name: result
        for name, result in candidate_results.items()
        if result["status"] == "completed"
    }
    selection = None
    if partition == "dev-test" and len(completed_candidates) == len(spec["candidates"]):
        selection = _select_protocol(
            candidate_score_rows, tie_candidate=spec["tie_candidate"]
        )
    ranking = sorted(
        completed_candidates,
        key=lambda name: (
            -float(completed_candidates[name]["cross_method_macro_rouge"]),
            name,
        ),
    )
    summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": (
            "completed"
            if len(completed_candidates) == len(spec["candidates"])
            else "partial_failure"
        ),
        "test_split_accessed": False,
        "study_id": spec["prereg"]["study_id"],
        "dataset": spec["dataset_label"],
        "partition": partition,
        "rows": len(ordered_ids),
        "candidate_count": len(spec["candidates"]),
        "method_count": len(METHODS),
        "method_families": list(METHODS),
        "multiple_comparison_candidate_count": len(spec["candidates"]),
        "ranking": ranking,
        "candidates": candidate_results,
        "selection": selection,
    }
    _write_json(output_root / "study_summary.json", summary)

    dev_by_candidate = (
        {name: result for name, result in dev_summary["candidates"].items()}
        if dev_summary is not None
        else {}
    )
    selected_protocol = selection["selected_protocol"] if selection else None
    for candidate_name, result in candidate_results.items():
        completed = result["status"] == "completed"
        dev_score = (
            float(result["cross_method_macro_rouge"])
            if partition == "dev" and completed
            else (
                float(dev_by_candidate[candidate_name]["cross_method_macro_rouge"])
                if partition == "dev-test"
                and dev_by_candidate.get(candidate_name, {}).get("status") == "completed"
                else None
            )
        )
        dev_test_score = (
            float(result["cross_method_macro_rouge"])
            if partition == "dev-test" and completed
            else None
        )
        if partition == "dev":
            promoted = completed
            reason = (
                "preregistration sends every still-valid protocol to one dev-test evaluation"
                if completed
                else result.get("failure")
            )
        else:
            promoted = completed and candidate_name == selected_protocol
            reason = (
                selection["selection_reason"]
                if promoted and selection
                else (
                    "not selected by preregistered dev-test rule"
                    if completed and selection
                    else result.get("failure", "selection unavailable because a protocol failed")
                )
            )
        _append_search_log(
            {
                "logged_at_utc": _utc_now(),
                "study_id": spec["prereg"]["study_id"],
                "dataset": spec["dataset_label"],
                "partition": partition,
                "candidate": candidate_name,
                "candidate_hash": result["candidate_hash"],
                "config_path": result["resolved_config_path"],
                "config_hash": result["resolved_config_sha256"],
                "dev_score": dev_score,
                "dev_test_score": dev_test_score,
                "status": result["status"],
                "promoted": promoted,
                "reason": reason,
                "comparison_family_size": len(spec["candidates"]),
                "test_split_accessed": False,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(STUDIES))
    parser.add_argument("--partition", required=True, choices=("dev", "dev-test"))
    args = parser.parse_args()
    summary = run_study(args.dataset, args.partition)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "dataset": summary["dataset"],
                "partition": summary["partition"],
                "ranking": summary["ranking"],
                "selection": summary["selection"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
