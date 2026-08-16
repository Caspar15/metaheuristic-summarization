"""Fail-closed frozen Multi-News secondary official-test evaluation.

The score-free dry run validates the canonical test identity, the exact D3b
scientific configuration, the nine-system matrix, and both evaluators.  The
execute mode requires a separately pinned activation and may only resume exact
verified checkpoints.  Test results never authorize tuning.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from statistics import fmean
from typing import Any, Mapping, Sequence

import stanza
from stanza.pipeline.core import DownloadMethod

from scripts.audit import run_govreport_final_test as shared
from scripts.audit.run_govreport_official_evaluator import METRIC_NAMES
from src.baselines.cli import summarize_jsonl_baseline
from src.data.partitions import resolve_experiment_partition
from src.data.policy import sha256_file, validate_dataset_policy_request
from src.data.schemas import extract_references
from src.eval.feasibility import classify_feasibility_row
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.eval.rouge import DEFAULT_METRICS, rouge_scores
from src.pipeline.select_sentences import validate_experiment_request
from src.utils.io import load_yaml, read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE = Path("configs/preregistrations/multinews_final_execution_freeze_v1.json")
DEFAULT_ACTIVATION = Path(
    "configs/preregistrations/multinews_final_execution_activation_v1.json"
)
DEFAULT_DRY_RUN = Path("docs/research/evidence/multinews_final_dry_run_v1.json")
OUTPUT_ROOT = Path("runs_v2/multinews_final_test_v1")
EXPECTED_TEST_ROWS = 5621
RANDOM_SEEDS = shared.RANDOM_SEEDS
DETERMINISTIC_BASELINES = (
    "sbert_centroid",
    "sbert_mmr_lambda_0.7",
    "pacsum_sbert_P03",
    "pacsum_tfidf_P08",
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
    "pacsum_tfidf_P08",
    "sbert_centroid",
    "sbert_mmr_lambda_0.7",
    "pacsum_sbert_P03",
)
PRIMARY_COMPARATOR = "pacsum_tfidf_P08"
SCIENTIFIC_FIELDS = (
    "seed",
    "features",
    "representations",
    "compute_budget",
    "candidate_budget",
    "candidates",
    "selector",
    "objectives",
    "routes",
    "coverage_guard",
    "optimizer",
    "length_control",
)


def _configure_shared() -> None:
    shared.OUTPUT_ROOT = OUTPUT_ROOT
    shared.RANDOM_SEEDS = RANDOM_SEEDS
    shared.DETERMINISTIC_BASELINES = DETERMINISTIC_BASELINES
    shared.FAMILY_LABELS = FAMILY_LABELS
    shared._baseline_config = _baseline_config
    shared._baseline_method = _baseline_method
    shared._prediction_evidence = _prediction_evidence


def _scientific_view(config: Mapping[str, Any]) -> dict[str, Any]:
    return {field: config[field] for field in SCIENTIFIC_FIELDS}


def _baseline_config(base: Mapping[str, Any], label: str) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    if label == "pacsum_sbert_P03":
        config["baselines"]["pacsum"].update(
            {"beta": 0.0, "lambda_previous": -0.3, "lambda_following": 0.7}
        )
    elif label == "pacsum_tfidf_P08":
        config["baselines"]["pacsum"].update(
            {"beta": 0.0, "lambda_previous": -0.8, "lambda_following": 0.2}
        )
    elif label == "sbert_mmr_lambda_0.7":
        config["baselines"]["sbert_mmr"]["lambda_relevance"] = 0.7
    return config


def _baseline_method(label: str) -> str:
    if label.startswith("random_seed_"):
        return "random"
    return {
        "sbert_mmr_lambda_0.7": "sbert_mmr",
        "pacsum_sbert_P03": "pacsum_sbert",
        "pacsum_tfidf_P08": "pacsum_tfidf",
    }.get(label, label)


def _prediction_evidence(
    *, label: str, path: Path, elapsed: float, freeze_sha256: str
) -> dict[str, Any]:
    rows = 0
    feasible = 0
    lengths: list[int] = []
    for row in read_jsonl(str(path)):
        rows += 1
        feasible += int(classify_feasibility_row(row, assume_legacy_feasible=False)[0])
        lengths.append(len(str(row.get("summary", "")).split()))
    if rows != EXPECTED_TEST_ROWS:
        raise ValueError(f"{label} produced {rows} rows, expected {EXPECTED_TEST_ROWS}")
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
        "predictions_path": shared._relative(path),
        "predictions_sha256": sha256_file(str(path)),
        "selected_indices_sha256": shared._selected_indices_sha256(path),
        "freeze_manifest_sha256": freeze_sha256,
        "implementation_commit": shared._git_commit(),
        "test_split_accessed": True,
        "test_scores_observed": False,
    }


def _dataset_identity(
    freeze: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    config = load_yaml(str(REPO_ROOT / freeze["pins"]["final_config"]["path"]))
    canonical = REPO_ROOT / freeze["pins"]["canonical_test"]["path"]
    preflight = validate_dataset_policy_request(config, str(canonical), "test")
    if resolve_experiment_partition(config, preflight) is not None:
        raise ValueError("final test must not use a validation partition")
    ordered_ids: list[str] = []
    references: dict[str, str] = {}
    for row in read_jsonl(str(canonical)):
        row_id = str(row.get("id"))
        refs = extract_references(row)
        if len(refs) != 1 or row_id in references:
            raise ValueError(f"invalid official test identity/reference for {row_id!r}")
        ordered_ids.append(row_id)
        references[row_id] = refs[0]
    if len(ordered_ids) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"official test yielded {len(ordered_ids)} rows, expected {EXPECTED_TEST_ROWS}"
        )
    return config, ordered_ids, references


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
    predictions = shared._prediction_texts(prediction_path, ordered_ids)
    means, per = rouge_scores(
        predictions,
        [[references[row_id]] for row_id in ordered_ids],
        metrics=DEFAULT_METRICS,
        return_per_example=True,
    )
    rows = [
        {"id": row_id, **scores, "macro_rouge": fmean(scores.values())}
        for row_id, scores in zip(ordered_ids, per)
    ]
    per_path = root / "per_example.jsonl"
    shared._write_jsonl(per_path, rows)
    evidence = {
        "status": "completed",
        "system": label,
        "protocol": "internal_multisentence_lsum_secondary",
        "rows": len(rows),
        "metrics": {**means, "macro_rouge": fmean(means.values())},
        "per_example_path": shared._relative(per_path),
        "per_example_sha256": sha256_file(str(per_path)),
        "test_split_accessed": True,
        "test_scores_observed": True,
    }
    shared._write_json(evidence_path, evidence)
    return evidence


def _load_freeze(path: Path, expected_sha256: str) -> dict[str, Any]:
    _configure_shared()
    freeze = shared._load_pinned_json(path, expected_sha256, "final execution freeze")
    if freeze.get("status") != "frozen_before_test_predictions_or_scores":
        raise ValueError("final execution package is not score-blind frozen")
    if freeze.get("one_shot") is not True or freeze.get("official_test_rows") != EXPECTED_TEST_ROWS:
        raise ValueError("final execution one-shot row contract drifted")
    if tuple(freeze.get("random_seeds", [])) != RANDOM_SEEDS:
        raise ValueError("final execution random seeds drifted")
    if tuple(freeze.get("system_families", [])) != FAMILY_LABELS:
        raise ValueError("final execution system matrix drifted")
    for label, pin in freeze["pins"].items():
        shared._check_pin(pin, label, binary=bool(pin.get("binary")))
    config = load_yaml(str(REPO_ROOT / freeze["pins"]["final_config"]["path"]))
    validate_experiment_request(config, "test")
    if freeze.get("study_id") != "multinews-secondary-final-evaluation-v1":
        raise ValueError("Multi-News final study identity drifted")
    config = load_yaml(str(REPO_ROOT / freeze["pins"]["final_config"]["path"]))
    source = load_yaml(
        str(REPO_ROOT / freeze["pins"]["frozen_d3b_source_config"]["path"])
    )
    if _scientific_view(config) != _scientific_view(source):
        raise ValueError("final Multi-News scientific config differs from frozen D3b")
    if config["experiment"] != {"status": "final_test_only", "dataset": "Multi-News"}:
        raise ValueError("final Multi-News experiment role drifted")
    if config["optimizer"] != {"method": "greedy"}:
        raise ValueError("final Multi-News selector drifted")
    if config["length_control"]["min_words"] != 200 or config["length_control"]["max_words"] != 250:
        raise ValueError("final Multi-News length contract drifted")
    if set(config["compute_budget"]["enabled_routes"]) != {"lexical", "semantic", "graph"}:
        raise ValueError("final Multi-News route set drifted")
    return freeze


def dry_run(*, freeze_path: Path, freeze_sha256: str, evidence_path: Path) -> dict[str, Any]:
    if (REPO_ROOT / OUTPUT_ROOT).exists():
        raise ValueError("Multi-News final-test output exists before score-free dry run")
    freeze = _load_freeze(freeze_path, freeze_sha256)
    config, ordered_ids, references = _dataset_identity(freeze)
    if len(ordered_ids) != EXPECTED_TEST_ROWS or len(references) != EXPECTED_TEST_ROWS:
        raise ValueError("Multi-News canonical test identity drifted")
    official = freeze["official_evaluator"]
    for label in (
        "stanza_resources_json",
        "stanza_tokenize_model",
        "stanza_mwt_model",
        "rouge_script",
        "wordnet_db",
    ):
        shared._check_pin(official[label], f"official evaluator {label}", binary=True)
    if not Path(official["perl_path"]).is_file():
        raise ValueError("frozen Perl executable is missing")
    if version("stanza") != official["stanza_version"] or version("pyrouge") != official["pyrouge_version"]:
        raise ValueError("official evaluator dependency version drifted")
    result = {
        "evidence_schema_version": "1.0",
        "study_id": "multinews-secondary-final-evaluation-v1",
        "measured_at_utc": shared._utc_now(),
        "status": "passed_without_predictions_or_scores",
        "freeze_manifest_path": freeze_path.as_posix(),
        "freeze_manifest_sha256": freeze_sha256,
        "rows": len(ordered_ids),
        "ordered_ids_sha256": hashlib.sha256("\n".join(ordered_ids).encode()).hexdigest(),
        "references_present": len(references),
        "system_families": list(FAMILY_LABELS),
        "random_seeds": list(RANDOM_SEEDS),
        "implementation_commit": shared._git_commit(),
        "tracked_worktree_clean": shared._git_clean(),
        "test_predictions_generated": False,
        "test_scores_observed": False,
    }
    shared._write_json(REPO_ROOT / evidence_path, result)
    return result


def _analyze(
    results: Mapping[str, Mapping[str, Any]],
    ordered_ids: Sequence[str],
    *,
    protocol: str,
) -> dict[str, Any]:
    metrics = METRIC_NAMES if protocol == "official" else DEFAULT_METRICS
    values: dict[str, dict[str, list[float]]] = {}
    for label, evidence in results.items():
        rows = list(read_jsonl(str(REPO_ROOT / evidence["per_example_path"])))
        if [row["id"] for row in rows] != list(ordered_ids):
            raise ValueError(f"{protocol} result order drifted for {label}")
        values[label] = {
            metric: [float(row[metric]) for row in rows]
            for metric in (*metrics, "macro_rouge")
        }
    proposed, comparator = values["proposed"], values[PRIMARY_COMPARATOR]
    primary = paired_bootstrap_difference(
        proposed["macro_rouge"],
        comparator["macro_rouge"],
        n_resamples=100_000,
        seed=20260901,
    )
    components: dict[str, dict[str, Any]] = {}
    raw_component_p: dict[str, float] = {}
    for offset, metric in enumerate(metrics, start=1):
        result = paired_bootstrap_difference(
            proposed[metric],
            comparator[metric],
            n_resamples=100_000,
            seed=20260901 + offset,
        )
        components[metric] = result
        raw_component_p[metric] = float(result["p_value_two_sided"])
    adjusted_components = holm_adjust(raw_component_p)
    for metric in metrics:
        components[metric]["p_value_holm_3"] = adjusted_components[metric]
    secondary: dict[str, dict[str, Any]] = {}
    raw_secondary_p: dict[str, float] = {}
    endpoint = 0
    for label in FAMILY_LABELS:
        if label == "proposed":
            continue
        secondary[label] = {}
        for metric in (*metrics, "macro_rouge"):
            result = paired_bootstrap_difference(
                proposed[metric],
                values[label][metric],
                n_resamples=100_000,
                seed=20261901 + endpoint,
            )
            secondary[label][metric] = result
            raw_secondary_p[f"{label}:{metric}"] = float(result["p_value_two_sided"])
            endpoint += 1
    adjusted_secondary = holm_adjust(raw_secondary_p)
    for label, endpoint_results in secondary.items():
        for metric, result in endpoint_results.items():
            result["p_value_holm_32"] = adjusted_secondary[f"{label}:{metric}"]
    macro_superiority = (
        float(primary["mean_difference"]) > 0
        and float(primary["ci_lower"]) > 0
        and float(primary["p_value_two_sided"]) <= 0.05
    )
    return {
        "status": "completed",
        "study_id": "multinews-secondary-final-evaluation-v1",
        "measured_at_utc": shared._utc_now(),
        "protocol": protocol,
        "partition": "Multi-News official test",
        "rows": len(ordered_ids),
        "corpus_ranking": sorted(
            ({"system": label, **dict(evidence["metrics"])} for label, evidence in results.items()),
            key=lambda row: (-float(row["macro_rouge"]), row["system"]),
        ),
        "primary_proposed_vs_pacsum_tfidf_P08": {
            "macro": primary,
            "components": components,
            "macro_superiority_supported": macro_superiority,
        },
        "secondary_holm_32": secondary,
        "decision": "report the frozen Multi-News secondary result without tuning",
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
    shared._load_activation(activation_path, activation_sha256, freeze_sha256)
    if not shared._git_clean():
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
        prediction_evidence[label] = shared._run_baseline(
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
        prediction_evidence[label] = shared._run_baseline(
            label=label,
            seed=seed,
            config=_baseline_config(config, label),
            canonical=canonical,
            preflight=preflight,
            root=output / "predictions" / label,
            freeze_sha256=freeze_sha256,
        )
        print(json.dumps({"prediction": label, "status": "completed"}))
    prediction_evidence["proposed"] = shared._run_proposed(
        config=config,
        canonical=canonical,
        ordered_ids=ordered_ids,
        root=output / "predictions/proposed",
        workers=workers,
        freeze_sha256=freeze_sha256,
    )
    proposed_evidence_path = output / "predictions/proposed/prediction_evidence.json"
    proposed_evidence = dict(prediction_evidence["proposed"])
    proposed_evidence["method"] = "provenance_aware_multiroute_greedy"
    shared._write_json(proposed_evidence_path, proposed_evidence)
    prediction_evidence["proposed"] = proposed_evidence
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
    shared._tokenize_files(
        nlp,
        [references[row_id] for row_id in ordered_ids],
        reference_plain,
        "ref",
    )
    shared._prepare_rouge_directory(reference_plain, reference_rouge, len(ordered_ids))
    alias_root, alias_created = shared._ensure_ascii_alias(ascii_alias)
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
            official_results[label] = shared._official_evaluate(
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
        label: value for label, value in official_results.items() if label.startswith("random_seed_")
    }
    random_internal = {
        label: value for label, value in internal_results.items() if label.startswith("random_seed_")
    }
    official_results["random_mean_10_seeds"] = shared._aggregate_random(
        protocol="official", seed_results=random_official, ordered_ids=ordered_ids
    )
    internal_results["random_mean_10_seeds"] = shared._aggregate_random(
        protocol="internal", seed_results=random_internal, ordered_ids=ordered_ids
    )
    family_official = {label: official_results[label] for label in FAMILY_LABELS}
    family_internal = {label: internal_results[label] for label in FAMILY_LABELS}
    official_analysis = _analyze(family_official, ordered_ids, protocol="official")
    internal_analysis = _analyze(family_internal, ordered_ids, protocol="internal")
    shared._write_json(output / "analysis.json", official_analysis)
    shared._write_json(output / "internal_analysis.json", internal_analysis)
    execution = {
        "status": "completed",
        "study_id": "multinews-secondary-final-evaluation-v1",
        "completed_at_utc": shared._utc_now(),
        "freeze_manifest_path": freeze_path.as_posix(),
        "freeze_manifest_sha256": freeze_sha256,
        "activation_path": activation_path.as_posix(),
        "activation_sha256": activation_sha256,
        "implementation_commit": shared._git_commit(),
        "scientific_code_commit": freeze["scientific_code_commit"],
        "workers": workers,
        "platform": platform.platform(),
        "python": sys.version,
        "dependencies": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "torch",
                "transformers",
                "sentence-transformers",
                "stanza",
                "pyrouge",
                "rouge-score",
            )
        },
        "analysis_path": shared._relative(output / "analysis.json"),
        "analysis_sha256": sha256_file(str(output / "analysis.json")),
        "internal_analysis_path": shared._relative(output / "internal_analysis.json"),
        "internal_analysis_sha256": sha256_file(str(output / "internal_analysis.json")),
        "decision": official_analysis["decision"],
        "test_split_accessed": True,
        "test_scores_observed": True,
        "post_score_tuning_permitted": False,
    }
    shared._write_json(output / "execution_evidence.json", execution)
    return official_analysis


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
                / ("attempt_" + shared.datetime.now(shared.timezone.utc).strftime("%Y%m%dT%H%M%SZ_failed"))
            )
            shared._write_json(
                attempt / "evidence.json",
                {
                    "status": "failed",
                    "failed_at_utc": shared._utc_now(),
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
