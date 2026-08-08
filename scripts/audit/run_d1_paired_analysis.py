"""Score preregistered D1 dev runs per row and run paired bootstrap tests.

There is no split CLI.  Both primary datasets and the exact comparison set are
fixed by the preregistration.  The script requires every prediction artifact
to contain exactly the frozen dev manifest IDs and never reads dev-test/test.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from scripts.audit.run_greedy_sensitivity import REPO_ROOT, STUDIES
from scripts.audit.run_length_contract_study import _git_commit, _utc_now, _write_json
from src.data.partitions import selected_ids_sha256
from src.data.policy import sha256_file
from src.data.schemas import extract_references
from src.eval.paired import holm_adjust, paired_bootstrap_difference
from src.eval.rouge import rouge_scores
from src.utils.io import read_jsonl


PREREGISTRATION = Path("configs/preregistrations/d1_paired_analysis_v1.json")
OUTPUT_ROOT = Path("runs_v2/d1_paired_analysis_v1")
METRICS = ("rouge1", "rouge2", "rougeLsum")

RUNS: dict[str, dict[str, str]] = {
    "multinews": {
        "s02b": "runs_v2/d1_three_route_followup/multinews/dev/S02b_three_route_capacity_80/greedy/run/predictions.jsonl",
        "without_semantic": "runs_v2/d1_capacity_matched_route_ablation/multinews/dev/A01_without_semantic/greedy/run/predictions.jsonl",
        "without_graph": "runs_v2/d1_capacity_matched_route_ablation/multinews/dev/A02_without_graph/greedy/run/predictions.jsonl",
        "lead": "runs_v2/a1_length_contract/multinews/dev/legacy_floor_200_cap_250/lead/run/predictions.jsonl",
        "random": "runs_v2/a1_length_contract/multinews/dev/legacy_floor_200_cap_250/random/run/predictions.jsonl",
    },
    "govreport": {
        "s02b": "runs_v2/d1_three_route_followup/govreport/dev/S02b_three_route_capacity_80/greedy/run/predictions.jsonl",
        "without_semantic": "runs_v2/d1_capacity_matched_route_ablation/govreport/dev/A01_without_semantic/greedy/run/predictions.jsonl",
        "without_graph": "runs_v2/d1_capacity_matched_route_ablation/govreport/dev/A02_without_graph/greedy/run/predictions.jsonl",
        "lead": "runs_v2/a1_length_contract/govreport/dev/dev_iqr_band_500_650/lead/run/predictions.jsonl",
        "random": "runs_v2/a1_length_contract/govreport/dev/dev_iqr_band_500_650/random/run/predictions.jsonl",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_preregistration() -> dict[str, Any]:
    value = _load_json(REPO_ROOT / PREREGISTRATION)
    if value.get("partition") != "dev" or value.get("dev_test_access") != "none":
        raise ValueError("paired analysis must remain bound to dev only")
    if not value.get("test_split_prohibited"):
        raise ValueError("paired analysis must prohibit test")
    if tuple(value.get("metrics", ())) != METRICS:
        raise ValueError("paired-analysis metric contract drift")
    bootstrap = value.get("bootstrap", {})
    if bootstrap.get("resamples") != 10_000 or bootstrap.get("seed") != 20_260_808:
        raise ValueError("paired-analysis bootstrap contract drift")
    return value


def _load_frozen_gold(dataset: str) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    spec = STUDIES[dataset]
    manifest_path = REPO_ROOT / spec["manifest"]
    if sha256_file(str(manifest_path)) != spec["manifest_sha256"]:
        raise ValueError(f"{dataset} manifest SHA-256 drift")
    manifest = _load_json(manifest_path)
    partition = manifest["partitions"]["dev"]
    ordered_ids = list(partition["selected_ids"])
    if selected_ids_sha256(ordered_ids) != partition["selected_ids_sha256"]:
        raise ValueError(f"{dataset} selected-ID digest drift")
    wanted = set(ordered_ids)
    references_by_id: dict[str, list[str]] = {}
    for row in read_jsonl(str(REPO_ROOT / spec["input"])):
        row_id = row.get("id")
        if row_id not in wanted:
            continue
        if row_id in references_by_id:
            raise ValueError(f"duplicate gold ID {row_id!r}")
        references = extract_references(row)
        if not references:
            raise ValueError(f"gold row {row_id!r} has no reference")
        references_by_id[row_id] = references
    if set(references_by_id) != wanted:
        missing = sorted(wanted - set(references_by_id))
        raise ValueError(f"{dataset} gold missing frozen IDs: {missing[:5]}")
    return ordered_ids, [references_by_id[row_id] for row_id in ordered_ids], {
        "manifest_path": spec["manifest"],
        "manifest_sha256": spec["manifest_sha256"],
        "selected_ids_sha256": partition["selected_ids_sha256"],
        "input_path": spec["input"],
        "input_sha256": sha256_file(str(REPO_ROOT / spec["input"])),
    }


def _load_predictions(path: Path, ordered_ids: list[str]) -> tuple[list[str], dict[str, Any]]:
    rows = list(read_jsonl(str(path)))
    prediction_by_id: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=1):
        row_id = row.get("id")
        summary = row.get("summary")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"{path}: invalid ID at row {line_number}")
        if row_id in prediction_by_id:
            raise ValueError(f"{path}: duplicate ID {row_id!r}")
        if not isinstance(summary, str):
            raise ValueError(f"{path}: non-string summary for {row_id!r}")
        prediction_by_id[row_id] = summary
    expected = set(ordered_ids)
    actual = set(prediction_by_id)
    if actual != expected:
        raise ValueError(
            f"{path}: prediction IDs differ from frozen dev; "
            f"missing={sorted(expected-actual)[:5]}, extra={sorted(actual-expected)[:5]}"
        )
    return [prediction_by_id[row_id] for row_id in ordered_ids], {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(str(path)),
        "rows": len(rows),
    }


def _score_dataset(dataset: str, output_root: Path) -> dict[str, Any]:
    ordered_ids, references, provenance = _load_frozen_gold(dataset)
    dataset_root = output_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=False)
    scores_by_label: dict[str, list[dict[str, float]]] = {}
    run_provenance: dict[str, Any] = {}
    for label, relative_path in RUNS[dataset].items():
        predictions, evidence = _load_predictions(REPO_ROOT / relative_path, ordered_ids)
        means, per_example = rouge_scores(
            predictions,
            references,
            metrics=METRICS,
            return_per_example=True,
        )
        rows = [
            {"id": row_id, **score}
            for row_id, score in zip(ordered_ids, per_example)
        ]
        label_root = dataset_root / label
        label_root.mkdir(parents=True, exist_ok=False)
        per_example_path = label_root / "per_example.jsonl"
        with per_example_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        evidence["per_example_path"] = per_example_path.relative_to(REPO_ROOT).as_posix()
        evidence["per_example_sha256"] = sha256_file(str(per_example_path))
        evidence["means"] = means
        run_provenance[label] = evidence
        scores_by_label[label] = rows
    return {
        "dataset": STUDIES[dataset]["dataset_label"],
        "partition": "dev",
        "rows": len(ordered_ids),
        "dataset_provenance": provenance,
        "runs": run_provenance,
        "scores": scores_by_label,
    }


def run() -> dict[str, Any]:
    preregistration = _load_preregistration()
    output_root = REPO_ROOT / OUTPUT_ROOT
    if output_root.exists():
        raise ValueError(f"refusing to overwrite paired analysis: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    scored = {
        dataset: _score_dataset(dataset, output_root)
        for dataset in ("multinews", "govreport")
    }

    raw_p_by_family: dict[str, dict[str, float]] = {
        "route": {},
        "cheap_baseline": {},
    }
    comparisons: dict[str, dict[str, Any]] = {
        "route": {},
        "cheap_baseline": {},
    }
    base_seed = int(preregistration["bootstrap"]["seed"])
    n_resamples = int(preregistration["bootstrap"]["resamples"])
    test_index = 0
    for family in ("route", "cheap_baseline"):
        for dataset in ("multinews", "govreport"):
            rows_by_label = scored[dataset]["scores"]
            for comparison in preregistration["comparisons"][family]:
                system = comparison["system"]
                reference = comparison["reference"]
                comparison_key = f"{dataset}:{system}_vs_{reference}"
                metric_results: dict[str, Any] = {}
                for metric in METRICS:
                    key = f"{comparison_key}:{metric}"
                    result = paired_bootstrap_difference(
                        [row[metric] for row in rows_by_label[system]],
                        [row[metric] for row in rows_by_label[reference]],
                        n_resamples=n_resamples,
                        seed=base_seed + test_index,
                    )
                    raw_p_by_family[family][key] = float(result["p_value_two_sided"])
                    metric_results[metric] = result
                    test_index += 1
                comparisons[family][comparison_key] = {
                    "interpretation": comparison["interpretation"],
                    "metrics": metric_results,
                }

    opportunities = {
        "route": int(
            preregistration["multiplicity"][
                "selection_aware_route_bonferroni_opportunities"
            ]
        ),
        "cheap_baseline": int(
            preregistration["multiplicity"][
                "selection_aware_baseline_bonferroni_opportunities"
            ]
        ),
    }
    for family in comparisons:
        adjusted = holm_adjust(raw_p_by_family[family])
        for comparison_key, comparison in comparisons[family].items():
            for metric, result in comparison["metrics"].items():
                key = f"{comparison_key}:{metric}"
                raw_p = float(result["p_value_two_sided"])
                result["p_value_holm"] = adjusted[key]
                result["p_value_selection_bonferroni"] = min(
                    1.0, raw_p * opportunities[family]
                )
                result["strong_endpoint_win"] = bool(
                    result["mean_difference"] > 0.0
                    and result["ci_lower"] > 0.0
                    and result["p_value_holm"] < 0.05
                    and result["p_value_selection_bonferroni"] < 0.05
                )

    summary = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "status": "completed_dev_only_not_promotion",
        "study_id": preregistration["study_id"],
        "preregistration_path": PREREGISTRATION.as_posix(),
        "preregistration_sha256": sha256_file(str(REPO_ROOT / PREREGISTRATION)),
        "implementation_commit": _git_commit(),
        "partition": "dev",
        "dev_test_accessed": False,
        "test_split_accessed": False,
        "bootstrap": preregistration["bootstrap"],
        "multiplicity": preregistration["multiplicity"],
        "datasets": {
            dataset: {
                key: value
                for key, value in dataset_result.items()
                if key != "scores"
            }
            for dataset, dataset_result in scored.items()
        },
        "comparisons": comparisons,
        "environment": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "rouge-score": importlib.metadata.version("rouge-score"),
        },
        "interpretation": (
            "Paired frozen-dev evidence only. Strong baselines and greedy reference "
            "remain incomplete, so this analysis cannot authorize dev-test or test."
        ),
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    summary = run()
    print(json.dumps(summary["comparisons"], ensure_ascii=False))


if __name__ == "__main__":
    main()
