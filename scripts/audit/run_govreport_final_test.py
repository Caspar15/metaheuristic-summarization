"""Fail-closed one-shot GovReport official-test runner.

``dry-run`` verifies every frozen identity and dependency without generating a
prediction or score. ``execute`` requires a separately pinned activation file,
then runs the nine frozen system families, official Stanza + Perl ROUGE, the
secondary internal evaluator, and preregistered paired inference. Interrupted
execution may resume only from artifacts whose freeze identity still matches.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

import stanza
from stanza.pipeline.core import DownloadMethod
import yaml

from scripts.audit.run_d3a_router_fusion_full_dev import _bounded_rows
from scripts.audit.run_govreport_official_evaluator import (
    METRIC_NAMES,
    _prepare_rouge_directory,
    _run_perl,
    _tokenize_files,
    parse_rouge_output,
)
from src.baselines.cli import summarize_jsonl_baseline
from src.data.partitions import resolve_experiment_partition
from src.data.policy import sha256_binary_file, sha256_file, validate_dataset_policy_request
from src.data.schemas import extract_references
from src.eval.feasibility import classify_feasibility_row
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.eval.rouge import DEFAULT_METRICS, rouge_scores
from src.pipeline.select_sentences import validate_experiment_request
from src.utils.io import load_yaml, read_jsonl, write_jsonl_atomic


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE = Path("configs/preregistrations/govreport_final_execution_freeze_v1.json")
DEFAULT_ACTIVATION = Path(
    "configs/preregistrations/govreport_final_execution_activation_v1.json"
)
DEFAULT_DRY_RUN = Path("docs/research/evidence/govreport_final_dry_run_v1.json")
OUTPUT_ROOT = Path("runs_v2/govreport_final_test_v1")
RANDOM_SEEDS = (3407, 2024, 42, 1337, 2026, 20260810, 7, 17, 23, 101)
DETERMINISTIC_BASELINES = (
    "sbert_centroid",
    "sbert_mmr_lambda_0.9",
    "pacsum_sbert_beta_0.5",
    "pacsum_tfidf_P07",
    "textrank",
    "lexrank",
    "lead",
)
FAMILY_LABELS = (
    "proposed",
    "lead",
    "random_mean_10_seeds",
    "textrank",
    "lexrank",
    "pacsum_tfidf_P07",
    "sbert_centroid",
    "sbert_mmr_lambda_0.9",
    "pacsum_sbert_beta_0.5",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl_atomic(str(path), (dict(row) for row in rows))


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _git_clean() -> bool:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
    )
    return not output.strip()


def _load_pinned_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    resolved = REPO_ROOT / path
    actual = sha256_file(str(resolved))
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA-256 is {actual}, expected {expected_sha256}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _check_pin(pin: Mapping[str, Any], label: str, *, binary: bool = False) -> None:
    path = REPO_ROOT / str(pin["path"])
    actual = sha256_binary_file(str(path)) if binary else sha256_file(str(path))
    expected = str(pin["sha256"])
    if actual != expected:
        raise ValueError(f"{label} SHA-256 is {actual}, expected {expected}")


def _load_freeze(path: Path, expected_sha256: str) -> dict[str, Any]:
    freeze = _load_pinned_json(path, expected_sha256, "final execution freeze")
    if freeze.get("status") != "frozen_before_test_predictions_or_scores":
        raise ValueError("final execution package is not score-blind frozen")
    if freeze.get("one_shot") is not True or freeze.get("official_test_rows") != 973:
        raise ValueError("final execution one-shot row contract drifted")
    if tuple(freeze.get("random_seeds", [])) != RANDOM_SEEDS:
        raise ValueError("final execution random seeds drifted")
    if tuple(freeze.get("system_families", [])) != FAMILY_LABELS:
        raise ValueError("final execution system matrix drifted")
    for label, pin in freeze["pins"].items():
        _check_pin(pin, label, binary=bool(pin.get("binary")))
    config = load_yaml(str(REPO_ROOT / freeze["pins"]["final_config"]["path"]))
    validate_experiment_request(config, "test")
    return freeze


def _load_activation(path: Path, expected_sha256: str, freeze_sha256: str) -> dict[str, Any]:
    activation = _load_pinned_json(path, expected_sha256, "final execution activation")
    if activation.get("status") != "authorized_for_one_shot_execution":
        raise ValueError("final execution is not activated")
    if activation.get("freeze_manifest_sha256") != freeze_sha256:
        raise ValueError("activation points to a different freeze package")
    dry = activation.get("score_free_dry_run", {})
    _check_pin(dry, "score-free dry run")
    evidence = json.loads((REPO_ROOT / dry["path"]).read_text(encoding="utf-8"))
    if evidence.get("status") != "passed_without_predictions_or_scores":
        raise ValueError("score-free dry run did not pass")
    return activation


def _dataset_identity(freeze: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    config = load_yaml(str(REPO_ROOT / freeze["pins"]["final_config"]["path"]))
    canonical = REPO_ROOT / freeze["pins"]["canonical_test"]["path"]
    preflight = validate_dataset_policy_request(config, str(canonical), "test")
    partition = resolve_experiment_partition(config, preflight)
    if partition is not None:
        raise ValueError("final test must not use a validation partition")
    ordered_ids: list[str] = []
    references: dict[str, str] = {}
    for row in read_jsonl(str(canonical)):
        row_id = str(row.get("id"))
        refs = extract_references(row)
        if len(refs) != 1:
            raise ValueError(f"official test requires one reference for {row_id!r}")
        if row_id in references:
            raise ValueError(f"duplicate official test ID {row_id!r}")
        ordered_ids.append(row_id)
        references[row_id] = refs[0]
    if len(ordered_ids) != 973:
        raise ValueError(f"official test yielded {len(ordered_ids)} rows, expected 973")
    return config, ordered_ids, references


def dry_run(
    *, freeze_path: Path, freeze_sha256: str, evidence_path: Path
) -> dict[str, Any]:
    if (REPO_ROOT / OUTPUT_ROOT).exists():
        raise ValueError("final-test output root already exists before dry run")
    freeze = _load_freeze(freeze_path, freeze_sha256)
    config, ordered_ids, references = _dataset_identity(freeze)
    if set(config["compute_budget"]["enabled_routes"]) != {"lexical", "semantic", "graph"}:
        raise ValueError("final proposed route set drifted")
    if config["optimizer"] != {"method": "mmr", "lambda_relevance": 0.7}:
        raise ValueError("final proposed selector drifted")
    if config["length_control"]["min_words"] != 500 or config["length_control"]["max_words"] != 650:
        raise ValueError("final length contract drifted")
    official = freeze["official_evaluator"]
    for label in ("stanza_resources_json", "stanza_tokenize_model", "stanza_mwt_model", "rouge_script", "wordnet_db"):
        _check_pin(official[label], f"official evaluator {label}", binary=True)
    perl = Path(official["perl_path"])
    if not perl.is_file():
        raise ValueError(f"frozen Perl executable is missing: {perl}")
    if version("stanza") != official["stanza_version"] or version("pyrouge") != official["pyrouge_version"]:
        raise ValueError("official evaluator dependency version drifted")
    result = {
        "evidence_schema_version": "1.0",
        "study_id": "govreport-centered-final-evaluation-v1",
        "measured_at_utc": _utc_now(),
        "status": "passed_without_predictions_or_scores",
        "freeze_manifest_path": freeze_path.as_posix(),
        "freeze_manifest_sha256": freeze_sha256,
        "rows": len(ordered_ids),
        "ordered_ids_sha256": hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest(),
        "references_present": len(references),
        "system_families": list(FAMILY_LABELS),
        "random_seeds": list(RANDOM_SEEDS),
        "implementation_commit": _git_commit(),
        "tracked_worktree_clean": _git_clean(),
        "test_predictions_generated": False,
        "test_scores_observed": False,
    }
    _write_json(REPO_ROOT / evidence_path, result)
    return result


def _baseline_config(base: Mapping[str, Any], label: str) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    if label == "pacsum_sbert_beta_0.5":
        config["baselines"]["pacsum"].update(
            {"beta": 0.5, "lambda_previous": 0.0, "lambda_following": 1.0}
        )
    elif label == "sbert_mmr_lambda_0.9":
        config["baselines"]["sbert_mmr"]["lambda_relevance"] = 0.9
    return config


def _baseline_method(label: str) -> str:
    if label.startswith("random_seed_"):
        return "random"
    return {
        "sbert_mmr_lambda_0.9": "sbert_mmr",
        "pacsum_sbert_beta_0.5": "pacsum_sbert",
        "pacsum_tfidf_P07": "pacsum_tfidf",
    }.get(label, label)


def _selected_indices_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for row in read_jsonl(str(path)):
        encoded = json.dumps(
            {"id": row["id"], "selected_indices": row["selected_indices"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(encoded + b"\n")
    return digest.hexdigest()


def _prediction_evidence(
    *, label: str, path: Path, elapsed: float, freeze_sha256: str
) -> dict[str, Any]:
    rows = 0
    feasible = 0
    lengths: list[int] = []
    for row in read_jsonl(str(path)):
        rows += 1
        classification = classify_feasibility_row(
            row, assume_legacy_feasible=False
        )
        feasible += int(classification[0])
        lengths.append(len(str(row.get("summary", "")).split()))
    if rows != 973:
        raise ValueError(f"{label} produced {rows} rows, expected 973")
    return {
        "status": "completed",
        "system": label,
        "rows": rows,
        "feasible_rows": feasible,
        "infeasible_rows": rows - feasible,
        "summary_words": {
            "mean": fmean(lengths),
            "min": min(lengths),
            "max": max(lengths),
        },
        "selection_seconds": elapsed,
        "predictions_path": _relative(path),
        "predictions_sha256": sha256_file(str(path)),
        "selected_indices_sha256": _selected_indices_sha256(path),
        "freeze_manifest_sha256": freeze_sha256,
        "implementation_commit": _git_commit(),
        "test_split_accessed": True,
        "test_scores_observed": False,
    }


def _run_baseline(
    *,
    label: str,
    seed: int | None,
    config: Mapping[str, Any],
    canonical: Path,
    preflight: Mapping[str, Any],
    root: Path,
    freeze_sha256: str,
) -> dict[str, Any]:
    evidence_path = root / "prediction_evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        predictions = REPO_ROOT / evidence["predictions_path"]
        if evidence.get("freeze_manifest_sha256") != freeze_sha256:
            raise ValueError(f"{label} checkpoint belongs to another freeze")
        if sha256_file(str(predictions)) != evidence["predictions_sha256"]:
            raise ValueError(f"{label} checkpoint predictions drifted")
        return evidence
    root.mkdir(parents=True, exist_ok=True)
    resolved_path = root / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    predictions = root / "predictions.jsonl"
    method = _baseline_method(label)
    started = time.perf_counter()
    summarize_jsonl_baseline(
        str(canonical),
        str(predictions),
        config,
        "test",
        baseline=method,
        ordering="document_order" if method == "lead" else None,
        first_k=3 if method == "lead" else None,
        seed=seed,
        dataset_preflight=dict(preflight),
        partition_preflight=None,
    )
    evidence = _prediction_evidence(
        label=label,
        path=predictions,
        elapsed=time.perf_counter() - started,
        freeze_sha256=freeze_sha256,
    )
    evidence.update(
        {
            "method": method,
            "seed": seed,
            "config_path": _relative(resolved_path),
            "config_sha256": sha256_file(str(resolved_path)),
        }
    )
    _write_json(evidence_path, evidence)
    return evidence


def _run_proposed(
    *,
    config: Mapping[str, Any],
    canonical: Path,
    ordered_ids: Sequence[str],
    root: Path,
    workers: int,
    freeze_sha256: str,
) -> dict[str, Any]:
    evidence_path = root / "prediction_evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        predictions = REPO_ROOT / evidence["predictions_path"]
        if evidence.get("freeze_manifest_sha256") != freeze_sha256:
            raise ValueError("proposed checkpoint belongs to another freeze")
        if sha256_file(str(predictions)) != evidence["predictions_sha256"]:
            raise ValueError("proposed checkpoint predictions drifted")
        return evidence
    root.mkdir(parents=True, exist_ok=True)
    resolved_path = root / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    predictions = root / "predictions.jsonl"
    temporary = predictions.with_suffix(".jsonl.tmp")
    started = time.perf_counter()
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for position, prediction in enumerate(
            _bounded_rows(read_jsonl(str(canonical)), dict(config), workers=workers)
        ):
            if position >= len(ordered_ids) or prediction.get("id") != ordered_ids[position]:
                raise ValueError("proposed official-test prediction order drifted")
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    if count != len(ordered_ids):
        raise ValueError(f"proposed produced {count} rows, expected {len(ordered_ids)}")
    temporary.replace(predictions)
    evidence = _prediction_evidence(
        label="proposed",
        path=predictions,
        elapsed=time.perf_counter() - started,
        freeze_sha256=freeze_sha256,
    )
    evidence.update(
        {
            "method": "provenance_aware_multiroute_mmr",
            "workers": workers,
            "config_path": _relative(resolved_path),
            "config_sha256": sha256_file(str(resolved_path)),
        }
    )
    _write_json(evidence_path, evidence)
    return evidence


def _prediction_texts(path: Path, ordered_ids: Sequence[str]) -> list[str]:
    by_id: dict[str, str] = {}
    for row in read_jsonl(str(path)):
        row_id, summary = row.get("id"), row.get("summary")
        if not isinstance(row_id, str) or not isinstance(summary, str) or row_id in by_id:
            raise ValueError(f"invalid prediction row in {path}")
        by_id[row_id] = summary
    if set(by_id) != set(ordered_ids):
        raise ValueError(f"prediction IDs do not match official test: {path}")
    return [by_id[row_id] for row_id in ordered_ids]


def _ensure_ascii_alias(drive: str) -> tuple[Path, bool]:
    normalized = drive.rstrip("\\/")
    if not normalized.endswith(":") or len(normalized) != 2:
        raise ValueError("ASCII alias must be a drive letter such as R:")
    mappings = subprocess.check_output(["subst"], text=True, errors="replace")
    for line in mappings.splitlines():
        if line.upper().startswith(normalized.upper()):
            if str(REPO_ROOT.resolve()).casefold() not in line.casefold():
                raise ValueError(f"{normalized} is already mapped to another path")
            return Path(normalized + "/"), False
    subprocess.run(
        ["subst", normalized, str(REPO_ROOT.resolve())], check=True, capture_output=True
    )
    return Path(normalized + "/"), True


def _official_evaluate(
    *,
    label: str,
    prediction_path: Path,
    ordered_ids: Sequence[str],
    reference_rouge: Path,
    nlp: stanza.Pipeline,
    freeze: Mapping[str, Any],
    alias_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    root = REPO_ROOT / OUTPUT_ROOT / "official" / label
    evidence_path = root / "evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("prediction_sha256") != sha256_file(str(prediction_path)):
            raise ValueError(f"official checkpoint prediction drifted for {label}")
        per_path = REPO_ROOT / evidence["per_example_path"]
        if evidence.get("per_example_sha256") != sha256_file(str(per_path)):
            raise ValueError(f"official checkpoint per-example drifted for {label}")
        return evidence
    predictions = _prediction_texts(prediction_path, ordered_ids)
    plain = root / "tokenized_plain"
    rouge_system = root / "rouge_workspace/system"
    _tokenize_files(nlp, predictions, plain, "dec")
    _prepare_rouge_directory(plain, rouge_system, len(ordered_ids))
    official = freeze["official_evaluator"]
    rouge_home = REPO_ROOT / official["rouge_home"]
    config = root / "rouge_workspace/settings.xml"
    raw, stderr, elapsed, command = _run_perl(
        perl=Path(official["perl_path"]),
        rouge_home=alias_root / rouge_home.relative_to(REPO_ROOT),
        system_rouge=alias_root / rouge_system.relative_to(REPO_ROOT),
        reference_rouge=alias_root / reference_rouge.relative_to(REPO_ROOT),
        config_path=alias_root / config.relative_to(REPO_ROOT),
        detailed=True,
    )
    averages, per_eval = parse_rouge_output(raw)
    if len(per_eval) != len(ordered_ids):
        raise ValueError(f"official per-row count drifted for {label}")
    lexical_order = sorted(f"{index}.dec" for index in range(len(ordered_ids)))
    task_to_id = {
        task_id: ordered_ids[int(filename.removesuffix(".dec"))]
        for task_id, filename in enumerate(lexical_order, start=1)
    }
    rows = []
    for task_id in range(1, len(ordered_ids) + 1):
        scores = per_eval[task_id]
        if set(scores) != set(METRIC_NAMES):
            raise ValueError(f"official metric set drifted for {label}/{task_id}")
        rows.append(
            {
                "id": task_to_id[task_id],
                **scores,
                "macro_rouge": fmean(scores.values()),
            }
        )
    order = {row_id: index for index, row_id in enumerate(ordered_ids)}
    rows.sort(key=lambda row: order[row["id"]])
    per_path = root / "per_example.jsonl"
    _write_jsonl(per_path, rows)
    raw_path = root / "rouge_raw.txt"
    raw_path.write_text(raw, encoding="utf-8", newline="\n")
    (root / "rouge_stderr.txt").write_text(stderr, encoding="utf-8", newline="\n")
    smoke_result = None
    if smoke:
        no_d = root / "rouge_workspace/settings_no_d.xml"
        aggregate, _, _, _ = _run_perl(
            perl=Path(official["perl_path"]),
            rouge_home=alias_root / rouge_home.relative_to(REPO_ROOT),
            system_rouge=alias_root / rouge_system.relative_to(REPO_ROOT),
            reference_rouge=alias_root / reference_rouge.relative_to(REPO_ROOT),
            config_path=alias_root / no_d.relative_to(REPO_ROOT),
            detailed=False,
        )
        aggregate_averages, _ = parse_rouge_output(aggregate)
        if aggregate_averages != averages:
            raise ValueError("official -d flag changed corpus scores")
        smoke_result = True
    evidence = {
        "status": "completed",
        "system": label,
        "rows": len(ordered_ids),
        "prediction_path": _relative(prediction_path),
        "prediction_sha256": sha256_file(str(prediction_path)),
        "metrics": {**averages, "macro_rouge": fmean(averages.values())},
        "per_example_path": _relative(per_path),
        "per_example_sha256": sha256_file(str(per_path)),
        "raw_output_path": _relative(raw_path),
        "raw_output_sha256": sha256_file(str(raw_path)),
        "execution_seconds": elapsed,
        "command": [str(value) for value in command],
        "d_flag_corpus_parity_smoke": smoke_result,
        "test_split_accessed": True,
        "test_scores_observed": True,
    }
    _write_json(evidence_path, evidence)
    return evidence


def _internal_evaluate(
    label: str,
    prediction_path: Path,
    ordered_ids: Sequence[str],
    references: Mapping[str, str],
) -> dict[str, Any]:
    root = REPO_ROOT / OUTPUT_ROOT / "internal" / label
    evidence_path = root / "evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        per_path = REPO_ROOT / evidence["per_example_path"]
        if evidence.get("per_example_sha256") != sha256_file(str(per_path)):
            raise ValueError(f"internal checkpoint per-example drifted for {label}")
        return evidence
    predictions = _prediction_texts(prediction_path, ordered_ids)
    means, per = rouge_scores(
        predictions,
        [[references[row_id]] for row_id in ordered_ids],
        metrics=DEFAULT_METRICS,
        return_per_example=True,
    )
    rows = [{"id": row_id, **scores} for row_id, scores in zip(ordered_ids, per)]
    per_path = root / "per_example.jsonl"
    _write_jsonl(per_path, rows)
    evidence = {
        "status": "completed",
        "system": label,
        "protocol": "internal_multisentence_lsum_secondary",
        "rows": len(rows),
        "metrics": {**means, "macro_rouge": fmean(means.values())},
        "per_example_path": _relative(per_path),
        "per_example_sha256": sha256_file(str(per_path)),
        "test_split_accessed": True,
        "test_scores_observed": True,
    }
    _write_json(evidence_path, evidence)
    return evidence


def _aggregate_random(
    *, protocol: str, seed_results: Mapping[str, Mapping[str, Any]], ordered_ids: Sequence[str]
) -> dict[str, Any]:
    metrics = (*METRIC_NAMES, "macro_rouge") if protocol == "official" else (*DEFAULT_METRICS, "macro_rouge")
    seed_rows: list[dict[str, dict[str, float]]] = []
    for evidence in seed_results.values():
        rows = list(read_jsonl(str(REPO_ROOT / evidence["per_example_path"])))
        seed_rows.append({row["id"]: {m: float(row[m]) for m in metrics} for row in rows})
    aggregated = []
    for row_id in ordered_ids:
        aggregated.append(
            {"id": row_id, **{metric: fmean(seed[row_id][metric] for seed in seed_rows) for metric in metrics}}
        )
    root = REPO_ROOT / OUTPUT_ROOT / protocol / "random_mean_10_seeds"
    per_path = root / "per_example.jsonl"
    _write_jsonl(per_path, aggregated)
    corpus = {
        metric: fmean(float(evidence["metrics"][metric]) for evidence in seed_results.values())
        for metric in metrics
    }
    result = {
        "status": "completed",
        "system": "random_mean_10_seeds",
        "protocol": protocol,
        "rows": len(ordered_ids),
        "seeds": list(RANDOM_SEEDS),
        "metrics": corpus,
        "per_example_path": _relative(per_path),
        "per_example_sha256": sha256_file(str(per_path)),
        "test_split_accessed": True,
        "test_scores_observed": True,
    }
    _write_json(root / "evidence.json", result)
    return result


def _analyze_official(
    results: Mapping[str, Mapping[str, Any]], ordered_ids: Sequence[str]
) -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {}
    for label, evidence in results.items():
        rows = list(read_jsonl(str(REPO_ROOT / evidence["per_example_path"])))
        if [row["id"] for row in rows] != list(ordered_ids):
            raise ValueError(f"official result order drifted for {label}")
        values[label] = {
            metric: [float(row[metric]) for row in rows]
            for metric in (*METRIC_NAMES, "macro_rouge")
        }
    proposed, comparator = values["proposed"], values["sbert_mmr_lambda_0.9"]
    primary = paired_bootstrap_difference(
        proposed["macro_rouge"], comparator["macro_rouge"],
        n_resamples=100_000, seed=20260830,
    )
    components: dict[str, dict[str, Any]] = {}
    component_p: dict[str, float] = {}
    for offset, metric in enumerate(METRIC_NAMES, start=1):
        result = paired_bootstrap_difference(
            proposed[metric], comparator[metric],
            n_resamples=100_000, seed=20260830 + offset,
        )
        components[metric] = result
        component_p[metric] = float(result["p_value_two_sided"])
    adjusted_components = holm_adjust(component_p)
    for metric in METRIC_NAMES:
        components[metric]["p_value_holm_3"] = adjusted_components[metric]
    secondary: dict[str, dict[str, Any]] = {}
    secondary_p: dict[str, float] = {}
    endpoint = 0
    for label in FAMILY_LABELS:
        if label == "proposed":
            continue
        secondary[label] = {}
        for metric in (*METRIC_NAMES, "macro_rouge"):
            result = paired_bootstrap_difference(
                proposed[metric], values[label][metric],
                n_resamples=100_000, seed=20261830 + endpoint,
            )
            secondary[label][metric] = result
            secondary_p[f"{label}:{metric}"] = float(result["p_value_two_sided"])
            endpoint += 1
    adjusted = holm_adjust(secondary_p)
    for label, metrics in secondary.items():
        for metric, result in metrics.items():
            result["p_value_holm_32"] = adjusted[f"{label}:{metric}"]
    macro_pass = (
        float(primary["mean_difference"]) > 0
        and float(primary["ci_lower"]) > 0
        and float(primary["p_value_two_sided"]) <= 0.05
    )
    component_guard = all(float(result["ci_upper"]) >= 0 for result in components.values())
    passed = macro_pass and component_guard
    return {
        "status": "completed",
        "study_id": "govreport-centered-final-evaluation-v1",
        "measured_at_utc": _utc_now(),
        "partition": "GovReport official test",
        "rows": len(ordered_ids),
        "corpus_ranking": sorted(
            ({"system": label, **dict(evidence["metrics"])} for label, evidence in results.items()),
            key=lambda row: (-float(row["macro_rouge"]), row["system"]),
        ),
        "primary_proposed_vs_sbert_mmr_lambda_0.9": {
            "macro": primary,
            "components": components,
            "macro_pass": macro_pass,
            "component_guard_pass": component_guard,
            "final_confirmatory_pass": passed,
        },
        "secondary_holm_32": secondary,
        "decision": (
            "retain GovReport-scoped superiority claim"
            if passed
            else "revoke or downgrade GovReport superiority claim; no tuning or rerun"
        ),
        "test_split_accessed": True,
        "test_scores_observed": True,
    }


def execute(
    *,
    freeze_path: Path,
    freeze_sha256: str,
    activation_path: Path,
    activation_sha256: str,
    workers: int,
    ascii_alias: str,
) -> dict[str, Any]:
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    freeze = _load_freeze(freeze_path, freeze_sha256)
    activation = _load_activation(activation_path, activation_sha256, freeze_sha256)
    if not _git_clean():
        raise ValueError("tracked worktree must be clean for one-shot execution")
    config, ordered_ids, references = _dataset_identity(freeze)
    canonical = REPO_ROOT / freeze["pins"]["canonical_test"]["path"]
    preflight = validate_dataset_policy_request(config, str(canonical), "test")
    output = REPO_ROOT / OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    os.environ["META_SUM_EMBEDDING_CACHE_DIR"] = str(output / "_embedding_cache")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    prediction_evidence: dict[str, dict[str, Any]] = {}
    for label in DETERMINISTIC_BASELINES:
        prediction_evidence[label] = _run_baseline(
            label=label,
            seed=None,
            config=_baseline_config(config, label),
            canonical=canonical,
            preflight=preflight,
            root=output / "predictions" / label,
            freeze_sha256=freeze_sha256,
        )
        print(json.dumps({"prediction": label, "status": "completed"}))
    for seed in RANDOM_SEEDS:
        label = f"random_seed_{seed}"
        prediction_evidence[label] = _run_baseline(
            label=label,
            seed=seed,
            config=_baseline_config(config, label),
            canonical=canonical,
            preflight=preflight,
            root=output / "predictions" / label,
            freeze_sha256=freeze_sha256,
        )
        print(json.dumps({"prediction": label, "status": "completed"}))
    prediction_evidence["proposed"] = _run_proposed(
        config=config,
        canonical=canonical,
        ordered_ids=ordered_ids,
        root=output / "predictions/proposed",
        workers=workers,
        freeze_sha256=freeze_sha256,
    )
    print(json.dumps({"prediction": "proposed", "status": "completed"}))

    official_cfg = freeze["official_evaluator"]
    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,mwt",
        use_gpu=False,
        model_dir=str(REPO_ROOT / official_cfg["stanza_resources"]),
        download_method=DownloadMethod.NONE,
        verbose=False,
    )
    reference_plain = output / "official/_shared/reference_plain"
    reference_rouge = output / "official/_shared/reference_rouge"
    _tokenize_files(nlp, [references[row_id] for row_id in ordered_ids], reference_plain, "ref")
    _prepare_rouge_directory(reference_plain, reference_rouge, len(ordered_ids))
    alias_root, alias_created = _ensure_ascii_alias(ascii_alias)
    official_results: dict[str, dict[str, Any]] = {}
    internal_results: dict[str, dict[str, Any]] = {}
    try:
        evaluation_labels = [
            "proposed",
            *DETERMINISTIC_BASELINES,
            *(f"random_seed_{seed}" for seed in RANDOM_SEEDS),
        ]
        for index, label in enumerate(evaluation_labels):
            prediction_path = REPO_ROOT / prediction_evidence[label]["predictions_path"]
            official_results[label] = _official_evaluate(
                label=label,
                prediction_path=prediction_path,
                ordered_ids=ordered_ids,
                reference_rouge=reference_rouge,
                nlp=nlp,
                freeze=freeze,
                alias_root=alias_root,
                smoke=index == 0,
            )
            internal_results[label] = _internal_evaluate(
                label, prediction_path, ordered_ids, references
            )
            print(json.dumps({"evaluation": label, "status": "completed"}))
    finally:
        if alias_created:
            subprocess.run(["subst", ascii_alias.rstrip("\\/"), "/D"], check=False)

    random_official = {
        label: official_results[label]
        for label in official_results if label.startswith("random_seed_")
    }
    random_internal = {
        label: internal_results[label]
        for label in internal_results if label.startswith("random_seed_")
    }
    official_results["random_mean_10_seeds"] = _aggregate_random(
        protocol="official", seed_results=random_official, ordered_ids=ordered_ids
    )
    internal_results["random_mean_10_seeds"] = _aggregate_random(
        protocol="internal", seed_results=random_internal, ordered_ids=ordered_ids
    )
    family_official = {label: official_results[label] for label in FAMILY_LABELS}
    family_internal = {label: internal_results[label] for label in FAMILY_LABELS}
    analysis = _analyze_official(family_official, ordered_ids)
    _write_json(output / "analysis.json", analysis)
    _write_json(
        output / "internal_summary.json",
        {
            "status": "completed",
            "protocol": "internal_multisentence_lsum_secondary",
            "corpus_ranking": sorted(
                ({"system": label, **dict(value["metrics"])} for label, value in family_internal.items()),
                key=lambda row: (-float(row["macro_rouge"]), row["system"]),
            ),
            "test_scores_observed": True,
        },
    )
    execution = {
        "status": "completed",
        "study_id": "govreport-centered-final-evaluation-v1",
        "completed_at_utc": _utc_now(),
        "freeze_manifest_path": freeze_path.as_posix(),
        "freeze_manifest_sha256": freeze_sha256,
        "activation_path": activation_path.as_posix(),
        "activation_sha256": activation_sha256,
        "implementation_commit": _git_commit(),
        "scientific_code_commit": freeze["scientific_code_commit"],
        "workers": workers,
        "platform": platform.platform(),
        "python": sys.version,
        "dependencies": {
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "sentence-transformers", "stanza", "pyrouge", "rouge-score")
        },
        "prediction_evidence": {
            label: {
                "path": _relative(REPO_ROOT / OUTPUT_ROOT / "predictions" / label / "prediction_evidence.json"),
                "sha256": sha256_file(str(REPO_ROOT / OUTPUT_ROOT / "predictions" / label / "prediction_evidence.json")),
            }
            for label in prediction_evidence
        },
        "analysis_path": _relative(output / "analysis.json"),
        "analysis_sha256": sha256_file(str(output / "analysis.json")),
        "decision": analysis["decision"],
        "test_split_accessed": True,
        "test_scores_observed": True,
        "post_score_tuning_permitted": False,
    }
    _write_json(output / "execution_evidence.json", execution)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("dry-run", "execute"))
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--dry-run-evidence", type=Path, default=DEFAULT_DRY_RUN)
    parser.add_argument("--activation", type=Path, default=DEFAULT_ACTIVATION)
    parser.add_argument("--expected-activation-sha256")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ascii-alias", default="R:")
    args = parser.parse_args()
    try:
        if args.mode == "dry-run":
            result = dry_run(
                freeze_path=args.freeze_manifest,
                freeze_sha256=args.expected_freeze_sha256,
                evidence_path=args.dry_run_evidence,
            )
        else:
            if not args.expected_activation_sha256:
                raise SystemExit("--expected-activation-sha256 is required for execute")
            result = execute(
                freeze_path=args.freeze_manifest,
                freeze_sha256=args.expected_freeze_sha256,
                activation_path=args.activation,
                activation_sha256=args.expected_activation_sha256,
                workers=args.workers,
                ascii_alias=args.ascii_alias,
            )
    except Exception as error:
        if args.mode == "execute":
            attempt = (
                REPO_ROOT
                / OUTPUT_ROOT
                / "attempts"
                / (datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%SZ_failed"))
            )
            _write_json(
                attempt / "evidence.json",
                {
                    "status": "failed",
                    "failed_at_utc": _utc_now(),
                    "failure": f"{type(error).__name__}: {error}",
                    "freeze_manifest_path": args.freeze_manifest.as_posix(),
                    "expected_freeze_sha256": args.expected_freeze_sha256,
                    "activation_path": args.activation.as_posix(),
                    "expected_activation_sha256": args.expected_activation_sha256,
                    "scientific_changes_permitted": False,
                    "resume_policy": "exact checkpoint only",
                },
            )
        raise
    print(json.dumps({"status": result["status"], "decision": result.get("decision")}))


if __name__ == "__main__":
    main()
