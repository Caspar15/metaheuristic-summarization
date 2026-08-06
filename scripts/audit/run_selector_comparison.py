"""Run a frozen, matched Greedy/MMR/NSGA-II validation pilot.

This runner deliberately separates the tracked, reference-blind sample
manifest from scoring.  It validates the full governed input first, selects
exactly the frozen IDs, writes complete prediction artifacts, and then emits
paired per-document ROUGE and uncertainty estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from scripts.audit.freeze_selector_pilot import file_sha256
from src.data.policy import validate_dataset_policy_request
from src.data.schemas import extract_references
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.eval.rouge import rouge_scores
from src.models.extractive.encoder_rank import clear_encoder_cache
from src.pipeline.select_sentences import summarize_one, validate_requested_split
from src.utils.io import load_yaml, read_jsonl, set_global_seed, write_jsonl_atomic


METRICS = ("rouge1", "rouge2", "rougeLsum")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def load_frozen_rows(input_path: Path, manifest: Mapping[str, Any]) -> List[Dict]:
    """Load exactly the manifest rows and preserve its declared order."""

    if file_sha256(input_path) != manifest.get("input_sha256"):
        raise ValueError("input SHA-256 does not match the frozen pilot manifest")
    selected_ids = manifest.get("selected_ids")
    if not isinstance(selected_ids, list) or not selected_ids:
        raise ValueError("pilot manifest selected_ids must be a non-empty list")
    if len(selected_ids) != int(manifest.get("sample_size", -1)):
        raise ValueError("pilot manifest sample_size does not match selected_ids")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("pilot manifest contains duplicate selected IDs")
    expected_digest = hashlib.sha256(
        "\n".join(selected_ids).encode("utf-8")
    ).hexdigest()
    if expected_digest != manifest.get("selected_ids_sha256"):
        raise ValueError("pilot manifest selected ID digest does not match")

    wanted = set(selected_ids)
    rows_by_id: Dict[str, Dict] = {}
    input_rows = 0
    for row in read_jsonl(str(input_path)):
        input_rows += 1
        row_id = row.get("id")
        if row_id in wanted:
            if row_id in rows_by_id:
                raise ValueError(f"duplicate selected input id: {row_id}")
            rows_by_id[row_id] = row
    if input_rows != int(manifest.get("input_rows", -1)):
        raise ValueError("input row count does not match the frozen pilot manifest")
    missing = [row_id for row_id in selected_ids if row_id not in rows_by_id]
    if missing:
        raise ValueError(f"frozen pilot IDs missing from input: {missing[:5]}")
    return [rows_by_id[row_id] for row_id in selected_ids]


def _labels(methods: Iterable[str], nsga_seeds: Iterable[int]) -> List[tuple[str, str, int]]:
    labels = []
    seeds = list(nsga_seeds)
    for method in methods:
        normalized = method.lower()
        if normalized == "nsga2":
            if not seeds:
                raise ValueError("NSGA-II requires at least one declared seed")
            labels.extend(
                (f"nsga2_seed{seed}", "nsga2", int(seed)) for seed in seeds
            )
        else:
            labels.append((normalized, normalized, 0))
    if not labels:
        raise ValueError("at least one method is required")
    if len({label for label, _, _ in labels}) != len(labels):
        raise ValueError("comparison labels must be unique")
    return labels


def _row_statistics(predictions: List[Dict]) -> Dict[str, Any]:
    words = [row["selection_evaluation"]["selected_words"] for row in predictions]
    sentences = [
        row["selection_evaluation"]["selected_sentences"] for row in predictions
    ]
    return {
        "rows": len(predictions),
        "feasible_rows": sum(bool(row.get("feasible")) for row in predictions),
        "mean_selected_words": float(np.mean(words)),
        "median_selected_words": float(np.median(words)),
        "mean_selected_sentences": float(np.mean(sentences)),
        "median_selected_sentences": float(np.median(sentences)),
    }


def run_comparison(
    *,
    input_path: Path,
    config_path: Path,
    manifest_path: Path,
    output_dir: Path,
    methods: Iterable[str],
    nsga_seeds: Iterable[int],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    cfg = load_yaml(str(config_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The subset itself is diagnostic, but its parent input must still pass
    # the exact frozen full-split policy before any row is read for scoring.
    dataset_preflight = validate_dataset_policy_request(
        cfg, str(input_path), "validation"
    )
    rows = load_frozen_rows(input_path, manifest)
    for row in rows:
        validate_requested_split(row, "validation")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_specs = _labels(methods, nsga_seeds)
    per_example_by_label: Dict[str, List[Dict[str, float]]] = {}
    result_summaries: Dict[str, Dict[str, Any]] = {}
    fingerprints_by_id: Dict[str, Dict[str, Any]] = {}

    for label, method, method_seed in run_specs:
        method_cfg = deepcopy(cfg)
        method_cfg.setdefault("optimizer", {})["method"] = method
        if method == "nsga2":
            method_cfg["seed"] = method_seed
        set_global_seed(method_cfg.get("seed"))
        clear_encoder_cache()
        started = time.perf_counter()
        predictions = [summarize_one(row, method_cfg) for row in rows]
        elapsed = time.perf_counter() - started

        for prediction in predictions:
            row_id = prediction["id"]
            inputs = prediction["selector_inputs"]
            reference = fingerprints_by_id.setdefault(row_id, inputs)
            if inputs != reference:
                raise RuntimeError(
                    f"selector inputs differ for {row_id!r} in run {label!r}"
                )

        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = label_dir / "predictions.jsonl"
        write_jsonl_atomic(str(predictions_path), predictions)

        summaries = [prediction["summary"] for prediction in predictions]
        references = [extract_references(row) for row in rows]
        means, per_example = rouge_scores(
            summaries, references, metrics=METRICS, return_per_example=True
        )
        per_example_rows = [
            {"id": row["id"], **scores}
            for row, scores in zip(rows, per_example)
        ]
        write_jsonl_atomic(str(label_dir / "per_example.jsonl"), per_example_rows)
        label_summary = {
            "label": label,
            "method": method,
            "seed": method_cfg.get("seed"),
            "config_sha256": _json_sha256(method_cfg),
            "predictions_sha256": file_sha256(predictions_path),
            "time_seconds_cold_process_model_cache": elapsed,
            "rouge": means,
            **_row_statistics(predictions),
        }
        (label_dir / "metrics.json").write_text(
            json.dumps(label_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        per_example_by_label[label] = per_example_rows
        result_summaries[label] = label_summary

    comparisons: Dict[str, Dict[str, Any]] = {}
    raw_p_values: Dict[str, float] = {}
    reference_label = "greedy" if "greedy" in per_example_by_label else run_specs[0][0]
    for label, _, _ in run_specs:
        if label == reference_label:
            continue
        metric_results = {}
        for metric_index, metric in enumerate(METRICS):
            system_values = [row[metric] for row in per_example_by_label[label]]
            reference_values = [
                row[metric] for row in per_example_by_label[reference_label]
            ]
            result = paired_bootstrap_difference(
                system_values,
                reference_values,
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed + metric_index,
            )
            key = f"{label}_vs_{reference_label}:{metric}"
            raw_p_values[key] = float(result["p_value_two_sided"])
            metric_results[metric] = result
        comparisons[f"{label}_vs_{reference_label}"] = metric_results

    adjusted = holm_adjust(raw_p_values)
    for comparison, metrics in comparisons.items():
        for metric, result in metrics.items():
            result["p_value_holm"] = adjusted[f"{comparison}:{metric}"]

    repo = Path.cwd()
    summary = {
        "status": "diagnostic_validation_pilot_not_gate_result",
        "interpretation": (
            "Frozen reference-blind pilot. Directional evidence only; full governed "
            "Multi-News validation, multiple NSGA-II seeds, and the second primary "
            "dataset are still required."
        ),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": file_sha256(manifest_path),
        "input_path": input_path.as_posix(),
        "input_sha256": file_sha256(input_path),
        "config_path": config_path.as_posix(),
        "config_file_sha256": file_sha256(config_path),
        "sample_size": len(rows),
        "row_ids_sha256": manifest["selected_ids_sha256"],
        "matched_selector_inputs": True,
        "selector_input_fingerprints_sha256": _json_sha256(fingerprints_by_id),
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
            "holm_family": "all non-reference method x ROUGE metric pilot comparisons",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "pymoo": importlib.metadata.version("pymoo"),
        },
        "git": _git_state(repo),
        "dataset_preflight": dataset_preflight,
        "results": result_summaries,
        "paired_comparisons": comparisons,
        "runtime_note": (
            "Each label clears the process-global encoder cache, so its total includes "
            "model construction/loading plus representation, routing, and selection. "
            "Operating-system file caches may still be warm. Stage-specific timing is "
            "not inferred from these totals."
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--methods", nargs="+", default=["greedy", "mmr", "nsga2"]
    )
    parser.add_argument("--nsga_seeds", nargs="+", type=int, default=[2024])
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=20240806)
    args = parser.parse_args()

    summary = run_comparison(
        input_path=Path(args.input),
        config_path=Path(args.config),
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        methods=args.methods,
        nsga_seeds=args.nsga_seeds,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"status": summary["status"], "results": summary["results"]}, indent=2))


if __name__ == "__main__":
    main()
