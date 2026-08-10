"""Aggregate a frozen NSGA-II seed extension against matched references."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from scripts.audit.freeze_selector_pilot import file_sha256
from src.eval.paired import holm_adjust, paired_bootstrap_difference


METRICS = ("rouge1", "rouge2", "rougeLsum")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _index(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("row has no valid id")
        if row_id in result:
            raise ValueError(f"duplicate row id: {row_id}")
        result[row_id] = row
    return result


def selection_stability(
    selections_by_seed: Mapping[str, Mapping[str, Iterable[int]]]
) -> Dict[str, float | int]:
    labels = list(selections_by_seed)
    if len(labels) < 2:
        raise ValueError("selection stability requires at least two seeds")
    row_ids = set(selections_by_seed[labels[0]])
    if not row_ids:
        raise ValueError("selection stability requires at least one row")
    if any(set(selections_by_seed[label]) != row_ids for label in labels[1:]):
        raise ValueError("seed prediction IDs are not aligned")

    pairwise_jaccards = []
    all_identical = 0
    unique_counts = []
    for row_id in sorted(row_ids):
        sets = [set(selections_by_seed[label][row_id]) for label in labels]
        frozen = [tuple(sorted(values)) for values in sets]
        unique_counts.append(len(set(frozen)))
        all_identical += int(len(set(frozen)) == 1)
        for first, second in combinations(sets, 2):
            union = first | second
            pairwise_jaccards.append(1.0 if not union else len(first & second) / len(union))
    return {
        "seeds": len(labels),
        "rows": len(row_ids),
        "seed_pairs_per_row": len(list(combinations(labels, 2))),
        "mean_pairwise_jaccard": float(np.mean(pairwise_jaccards)),
        "median_pairwise_jaccard": float(np.median(pairwise_jaccards)),
        "all_seeds_identical_rows": all_identical,
        "all_seeds_identical_rate": all_identical / len(row_ids),
        "mean_unique_selection_sets_per_row": float(np.mean(unique_counts)),
        "max_unique_selection_sets_per_row": int(max(unique_counts)),
    }


def aggregate(
    *,
    greedy_dir: Path,
    mmr_dir: Path,
    seed_dirs: List[Path],
    output_path: Path,
    n_resamples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    greedy_scores = _index(_read_jsonl(greedy_dir / "per_example.jsonl"))
    row_order = list(greedy_scores)
    mmr_scores = _index(_read_jsonl(mmr_dir / "per_example.jsonl"))
    if set(mmr_scores) != set(greedy_scores):
        raise ValueError("MMR score IDs do not match Greedy")
    greedy_metrics = json.loads(
        (greedy_dir / "metrics.json").read_text(encoding="utf-8")
    )
    mmr_metrics = json.loads((mmr_dir / "metrics.json").read_text(encoding="utf-8"))
    greedy_predictions = _index(_read_jsonl(greedy_dir / "predictions.jsonl"))
    seed_artifacts = {}
    selection_sets = {}
    p_values = {}
    comparisons = {}
    selector_inputs_reference = {
        row_id: row["selector_inputs"] for row_id, row in greedy_predictions.items()
    }

    for seed_dir in seed_dirs:
        metrics = json.loads((seed_dir / "metrics.json").read_text(encoding="utf-8"))
        label = metrics["label"]
        scores = _index(_read_jsonl(seed_dir / "per_example.jsonl"))
        predictions = _index(_read_jsonl(seed_dir / "predictions.jsonl"))
        if set(scores) != set(greedy_scores) or set(predictions) != set(greedy_scores):
            raise ValueError(f"row IDs for {label} do not match Greedy")
        if any(
            predictions[row_id]["selector_inputs"] != selector_inputs_reference[row_id]
            for row_id in predictions
        ):
            raise ValueError(f"selector inputs for {label} do not match Greedy")

        metric_comparisons = {}
        for metric_index, metric in enumerate(METRICS):
            nsga_values = [scores[row_id][metric] for row_id in row_order]
            greedy_values = [
                greedy_scores[row_id][metric] for row_id in row_order
            ]
            result = paired_bootstrap_difference(
                nsga_values,
                greedy_values,
                n_resamples=n_resamples,
                seed=bootstrap_seed + metric_index,
            )
            key = f"{label}_vs_greedy:{metric}"
            p_values[key] = float(result["p_value_two_sided"])
            metric_comparisons[metric] = result
        comparisons[f"{label}_vs_greedy"] = metric_comparisons
        selection_sets[label] = {
            row_id: predictions[row_id]["selected_indices"] for row_id in predictions
        }
        seed_artifacts[label] = {
            "metrics": metrics,
            "metrics_sha256": file_sha256(seed_dir / "metrics.json"),
            "per_example_sha256": file_sha256(seed_dir / "per_example.jsonl"),
            "predictions_sha256": file_sha256(seed_dir / "predictions.jsonl"),
        }

    adjusted = holm_adjust(p_values)
    for comparison, metric_results in comparisons.items():
        for metric, result in metric_results.items():
            result["p_value_holm_15_test_family"] = adjusted[f"{comparison}:{metric}"]

    across_seeds = {}
    for metric in METRICS:
        values = [artifact["metrics"]["rouge"][metric] for artifact in seed_artifacts.values()]
        across_seeds[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "sample_standard_deviation": float(np.std(values, ddof=1)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
            "greedy": float(greedy_metrics["rouge"][metric]),
            "mmr": float(mmr_metrics["rouge"][metric]),
        }

    summary = {
        "status": "diagnostic_nsga2_pilot_stability_extension",
        "seeds": [artifact["metrics"]["seed"] for artifact in seed_artifacts.values()],
        "population_size": 64,
        "generations": 80,
        "matched_selector_inputs": True,
        "all_feasible": all(
            artifact["metrics"]["feasible_rows"] == artifact["metrics"]["rows"]
            for artifact in seed_artifacts.values()
        ),
        "across_seed_metrics": across_seeds,
        "selection_stability": selection_stability(selection_sets),
        "paired_vs_greedy": comparisons,
        "seed_artifacts": seed_artifacts,
        "reference_artifacts": {
            "greedy_predictions_sha256": file_sha256(greedy_dir / "predictions.jsonl"),
            "mmr_predictions_sha256": file_sha256(mmr_dir / "predictions.jsonl"),
        },
        "bootstrap": {
            "resamples": n_resamples,
            "seed": bootstrap_seed,
            "holm_family_size": len(p_values),
        },
        "interpretation": (
            "Report all seeds. This extension was frozen after seed 2024 was already "
            "observed, so it is stability evidence rather than a preregistered formal study."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy_dir", required=True)
    parser.add_argument("--mmr_dir", required=True)
    parser.add_argument("--seed_dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap_seed", type=int, default=20240806)
    args = parser.parse_args()
    summary = aggregate(
        greedy_dir=Path(args.greedy_dir),
        mmr_dir=Path(args.mmr_dir),
        seed_dirs=[Path(value) for value in args.seed_dirs],
        output_path=Path(args.output),
        n_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({
        "across_seed_metrics": summary["across_seed_metrics"],
        "selection_stability": summary["selection_stability"],
    }, indent=2))


if __name__ == "__main__":
    main()
