"""Validate and summarize one frozen-dev Gate 2 baseline family.

This audit deliberately exposes no partition argument.  It reads only the
preregistered ``dev`` family artifacts and never opens dev-test or test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.audit.run_gate2_baseline_matrix import STUDIES, load_protocol


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_NAME = "analysis_summary.json"
A1_LEAD = {
    "multinews": REPO_ROOT
    / "runs_v2/a1_length_contract/multinews/dev/legacy_floor_200_cap_250/lead/run",
    "govreport": REPO_ROOT
    / "runs_v2/a1_length_contract/govreport/dev/dev_iqr_band_500_650/lead/run",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank_results(results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for candidate, result in results.items():
        if result.get("status") != "completed":
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "candidate": candidate,
                "method": result["method"],
                "macro_rouge": metrics["macro_rouge"],
                "rouge": metrics["rouge"],
            }
        )
    return sorted(rows, key=lambda row: (-row["macro_rouge"], row["candidate"]))


def _compare_selections(
    reference_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    reference = {
        str(row["id"]): tuple(int(value) for value in row["selected_indices"])
        for row in reference_rows
    }
    candidate = {
        str(row["id"]): tuple(int(value) for value in row["selected_indices"])
        for row in candidate_rows
    }
    if reference.keys() != candidate.keys():
        raise ValueError("selection comparison ID sets differ")
    exact = 0
    jaccard_sum = 0.0
    for row_id in reference:
        left = reference[row_id]
        right = candidate[row_id]
        exact += left == right
        union = set(left) | set(right)
        jaccard_sum += len(set(left) & set(right)) / len(union) if union else 1.0
    rows = len(reference)
    return {
        "rows": rows,
        "exact_selected_indices_rows": exact,
        "exact_selected_indices_rate": exact / rows if rows else 1.0,
        "mean_selected_indices_jaccard": jaccard_sum / rows if rows else 1.0,
    }


def _prediction_diagnostics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    degenerate = sum(
        bool((row.get("baseline_diagnostics") or {}).get("score_degenerate"))
        for row in rows
    )
    return {
        "rows": len(rows),
        "score_degenerate_rows": degenerate,
        "score_degenerate_rate": degenerate / len(rows) if rows else 0.0,
    }


def summarize(dataset: str, family: str) -> dict[str, Any]:
    protocol = load_protocol()
    root = REPO_ROOT / "runs_v2/gate2_baseline_matrix_v1" / dataset / "dev" / family
    family_path = root / "family_summary.json"
    family_summary = json.loads(family_path.read_text(encoding="utf-8"))
    expected = int(protocol["expected_candidate_counts_per_dataset"][family])
    if family_summary["candidate_count"] != expected:
        raise ValueError("family candidate count differs from preregistration")
    if family_summary["completed_count"] != expected or family_summary["failed_count"]:
        raise ValueError("family is not complete; refusing a winner summary")
    if family_summary["partition"] != "dev":
        raise ValueError("Gate 2 family summary is not frozen dev")

    for candidate, result in family_summary["results"].items():
        if result.get("dev_test_accessed") is not False:
            raise ValueError(f"{candidate} does not prove dev-test non-access")
        if result.get("test_split_accessed") is not False:
            raise ValueError(f"{candidate} does not prove test non-access")
        evidence_path = root / candidate / result["method"] / "run/evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("partition") != "dev":
            raise ValueError(f"{candidate} evidence is not dev")
        if evidence.get("dev_test_accessed") is not False:
            raise ValueError(f"{candidate} evidence accessed dev-test")
        if evidence.get("test_split_accessed") is not False:
            raise ValueError(f"{candidate} evidence accessed test")

    ranking = _rank_results(family_summary["results"])
    winner = ranking[0]
    diagnostics: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    lead_root = A1_LEAD[dataset]
    lead_predictions = _load_jsonl(lead_root / "predictions.jsonl")

    for candidate in [winner["candidate"], "pacsum_tfidf_B10_beta_1.0"]:
        if candidate not in family_summary["results"]:
            continue
        result = family_summary["results"][candidate]
        predictions_path = root / candidate / result["method"] / "run/predictions.jsonl"
        predictions = _load_jsonl(predictions_path)
        diagnostics[candidate] = {
            **_prediction_diagnostics(predictions),
            "predictions_sha256": _sha256(predictions_path),
        }
        comparisons[f"{candidate}_vs_frozen_lead"] = _compare_selections(
            lead_predictions, predictions
        )

    lead_metrics = json.loads((lead_root / "metrics.json").read_text(encoding="utf-8"))
    lead_macro = sum(lead_metrics["rouge"].values()) / 3.0
    output = {
        "evidence_schema_version": "1.0",
        "measured_at_utc": _utc_now(),
        "study_id": "gate2-baseline-matrix-v1-family-analysis",
        "implementation_commit": _git_head(),
        "dataset": STUDIES[dataset]["dataset_label"],
        "partition": "dev",
        "family": family,
        "candidate_count": expected,
        "family_summary_path": family_path.relative_to(REPO_ROOT).as_posix(),
        "family_summary_sha256": _sha256(family_path),
        "preregistration_path": "configs/preregistrations/gate2_baseline_matrix_v1.json",
        "preregistration_sha256": family_summary["preregistration_sha256"],
        "winner": winner,
        "ranking": ranking,
        "frozen_lead": {"macro_rouge": lead_macro, "rouge": lead_metrics["rouge"]},
        "winner_minus_frozen_lead_macro": winner["macro_rouge"] - lead_macro,
        "prediction_diagnostics": diagnostics,
        "selection_comparisons": comparisons,
        "interpretation_guard": (
            "Ranking is descriptive frozen-dev evidence only. A score-degenerate "
            "PacSum endpoint is a canonical-order skip-tolerant control, not "
            "evidence that directed centrality contributes. No candidate is "
            "promoted until both families and both primary datasets are complete."
        ),
        "dev_test_accessed": False,
        "test_split_accessed": False,
    }
    output_path = root / OUTPUT_NAME
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(STUDIES), required=True)
    parser.add_argument("--family", choices=("non_plm", "plm"), required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(summarize(args.dataset, args.family), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
