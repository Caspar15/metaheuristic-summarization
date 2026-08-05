"""F-17/F-18: score N runs on the ID intersection so configs are paired.

WHY THIS EXISTS
---------------
Before F-17, an infeasible document made the whole run abort, so different
selector configs (and old ad-hoc runs predating the "feasible" field, e.g.
Caspar's 5,613 / 5,620 / 5,621-row artifacts) ended up with different sets of
documents actually present in predictions.jsonl. Averaging each run over its
own row count is not a fair comparison: a run that happened to skip its
hardest documents looks better for reasons that have nothing to do with
selection quality.

This script does not change any run's rows. It computes the ID intersection
across the given predictions.jsonl files, scores every run restricted to
that same intersection with the same gold subset, and writes per-document
ROUGE scores to disk -- the input a paired significance test (bootstrap,
Wilcoxon, ...) needs. That test is intentionally NOT run here; this script
only produces the paired, matched-denominator numbers it would consume.

Pre-F-17 artifacts do not contain enough information to prove row
feasibility. ``--assume-legacy-feasible`` is therefore an explicit,
unverified diagnostic assumption about the rows that happen to be present;
it does not certify them and it does not hide missing IDs. Missing legacy IDs
are listed in the report, while an incomplete post-F-17 artifact is rejected.

USAGE
-----
    python -m scripts.audit.paired_run_intersection \\
      --pred runs/greedy_mean/predictions.jsonl \\
             runs/greedy_length_normalized/predictions.jsonl \\
             runs/nsga2_mean/predictions.jsonl \\
      --gold data/processed/multi_news_validation_canonical.jsonl \\
      --out_dir runs/audit/f18_intersection_recompute \\
      --protocol multisentence_lsum \\
      --assume-legacy-feasible

Writes, under --out_dir:
    include_ids.json          sorted list of IDs scored (the intersection)
    intersection_report.json  hashes, infeasible reasons, missing legacy IDs,
                              and feasible IDs excluded for pairing
    <label>/per_example.jsonl one row per document: id, rouge1, rouge2, rougeLsum
    <label>/metrics.csv       corpus-mean metrics over the intersection only

<label> defaults to the basename of each --pred file's parent directory;
override with --labels if two runs share a parent directory name.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from typing import Any, Dict, List, Set

from src.data.schemas import extract_references
from src.eval.feasibility import classify_feasibility_row
from src.eval.protocol import MULTISENTENCE_LSUM
from src.eval.rouge import rouge_scores
from src.pipeline.evaluate import plan_feasibility_scoring
from src.utils.io import read_jsonl


def _default_label(pred_path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.abspath(pred_path))) or pred_path


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(pred_path: str, *, assume_legacy_feasible: bool) -> Dict:
    rows = list(read_jsonl(pred_path))
    pred_by_id: Dict[str, str] = {}
    infeasible_rows: List[Dict[str, Any]] = []
    for line_number, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(
                f"{pred_path}: prediction row {line_number} has no valid 'id'"
            )
        if row_id in pred_by_id:
            raise ValueError(f"{pred_path}: duplicate prediction id: {row_id}")
        summary = row.get("summary")
        if not isinstance(summary, str):
            raise ValueError(
                f"{pred_path}: prediction row {line_number} must contain a string 'summary'"
            )
        pred_by_id[row_id] = summary
        feasible, _ = classify_feasibility_row(
            row, assume_legacy_feasible=assume_legacy_feasible
        )
        if not feasible:
            infeasible_rows.append(
                {
                    "id": row_id,
                    "infeasible_code": row.get("infeasible_code"),
                    "infeasible_reason": row.get("infeasible_reason"),
                    "violations": row.get("violations"),
                }
            )
    include_ids, stats = plan_feasibility_scoring(
        rows, feasible_only=True, assume_legacy_feasible=assume_legacy_feasible
    )
    assert include_ids is not None
    return {
        "path": pred_path,
        "sha256": _sha256(pred_path),
        "pred_by_id": pred_by_id,
        "include_ids": include_ids,
        "infeasible": infeasible_rows,
        "is_legacy": stats["legacy_schema_assumed_feasible_rows"] > 0,
        "stats": stats,
    }


def _validate_run_against_gold(run: Dict, gold_ids: Set[str]) -> List[str]:
    """Reject unknown IDs and incomplete post-F-17 artifacts."""

    prediction_ids = set(run["pred_by_id"])
    unknown_ids = sorted(prediction_ids - gold_ids)
    if unknown_ids:
        raise ValueError(
            f"{run['path']}: {len(unknown_ids)} prediction ID(s) are not in "
            f"--gold; first: {unknown_ids[:5]}"
        )
    missing_ids = sorted(gold_ids - prediction_ids)
    if missing_ids and not run["is_legacy"]:
        raise ValueError(
            f"{run['path']}: post-F-17 artifact is incomplete; "
            f"{len(missing_ids)} gold ID(s) have no prediction; first: {missing_ids[:5]}"
        )
    return missing_ids


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pred", nargs="+", required=True,
                    help="two or more predictions.jsonl paths to compare")
    ap.add_argument("--gold", required=True, help="canonical gold dataset jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--protocol",
        required=True,
        choices=[MULTISENTENCE_LSUM],
        help="explicit evaluation protocol; this audit currently supports Multi-News/GovReport",
    )
    ap.add_argument("--labels", nargs="+", default=None,
                     help="one label per --pred, same order (default: parent dir name)")
    ap.add_argument(
        "--assume-legacy-feasible",
        action="store_true",
        help=(
            "required for a pre-F-17 artifact with no top-level 'feasible' "
            "field; explicitly makes the unverified diagnostic assumption "
            "that every recorded row is feasible. Missing IDs remain reported."
        ),
    )
    args = ap.parse_args()

    if len(args.pred) < 2:
        raise ValueError("need at least two --pred runs to compare")
    labels = args.labels or [_default_label(p) for p in args.pred]
    if len(labels) != len(args.pred):
        raise ValueError("--labels must have exactly one entry per --pred")
    if len(set(labels)) != len(labels):
        raise ValueError(f"--labels must be unique, got {labels}")

    runs = [
        _load_run(p, assume_legacy_feasible=args.assume_legacy_feasible)
        for p in args.pred
    ]

    common_ids: Set[str] = set(runs[0]["include_ids"])
    for run in runs[1:]:
        common_ids &= run["include_ids"]
    if not common_ids:
        raise ValueError("intersection of feasible IDs across --pred runs is empty")
    sorted_ids = sorted(common_ids)

    gold_by_id: Dict[str, List[str]] = {}
    for line_number, row in enumerate(read_jsonl(args.gold), start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"gold row {line_number} has no valid 'id'")
        if row_id in gold_by_id:
            raise ValueError(f"duplicate gold id: {row_id}")
        references_for_row = extract_references(row)
        if not references_for_row:
            raise ValueError(f"gold row {line_number} has no non-empty references")
        gold_by_id[row_id] = references_for_row

    gold_ids = set(gold_by_id)
    for run in runs:
        run["missing_prediction_ids"] = _validate_run_against_gold(run, gold_ids)

    missing_from_gold = [i for i in sorted_ids if i not in gold_by_id]
    if missing_from_gold:  # defensive; unknown IDs are rejected above
        raise ValueError(f"intersection IDs missing from gold: {missing_from_gold[:5]}")
    references = [gold_by_id[i] for i in sorted_ids]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(
        os.path.join(args.out_dir, "include_ids.json"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(sorted_ids, f, ensure_ascii=False, indent=2)

    intersection_report = {
        "evaluation_protocol": args.protocol,
        "gold_path": args.gold,
        "gold_sha256": _sha256(args.gold),
        "intersection_size": len(sorted_ids),
        "runs": [
            {
                "label": label,
                "path": run["path"],
                "sha256": run["sha256"],
                "total_rows": run["stats"]["total_rows"],
                "feasible_rows": run["stats"]["feasible_rows"],
                "legacy_schema_assumed_feasible_rows": run["stats"][
                    "legacy_schema_assumed_feasible_rows"
                ],
                "infeasible": run["infeasible"],
                "missing_prediction_ids": run["missing_prediction_ids"],
                "excluded_feasible_from_intersection": sorted(
                    run["include_ids"] - common_ids
                ),
            }
            for label, run in zip(labels, runs)
        ],
    }
    with open(
        os.path.join(args.out_dir, "intersection_report.json"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(intersection_report, f, ensure_ascii=False, indent=2)

    print(f"intersection size: {len(sorted_ids)}")
    for label, run in zip(labels, runs):
        excluded = len(run["include_ids"] - common_ids)
        print(
            f"  {label}: {run['stats']['total_rows']} rows on disk, "
            f"{run['stats']['feasible_rows']} feasible, "
            f"{excluded} excluded to reach the intersection"
        )

    print(f"\n{'run':36s} {'R-1':>8s} {'R-2':>8s} {'R-Lsum':>8s}")
    print("-" * 62)
    for label, run in zip(labels, runs):
        predictions = [run["pred_by_id"][i] for i in sorted_ids]
        means, per_example = rouge_scores(predictions, references, return_per_example=True)

        run_dir = os.path.join(args.out_dir, label)
        os.makedirs(run_dir, exist_ok=True)
        per_example_path = os.path.join(run_dir, "per_example.jsonl")
        with open(per_example_path, "w", encoding="utf-8", newline="\n") as f:
            for doc_id, scores in zip(sorted_ids, per_example):
                f.write(json.dumps({"id": doc_id, **scores}, ensure_ascii=False) + "\n")

        metrics_path = os.path.join(run_dir, "metrics.csv")
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["metric", "value"])
            w.writerow(["evaluation_protocol", args.protocol])
            w.writerow(["prediction_sha256", run["sha256"]])
            w.writerow(["gold_sha256", intersection_report["gold_sha256"]])
            w.writerow(["intersection_size", len(sorted_ids)])
            for metric, value in means.items():
                w.writerow([metric, f"{value:.6f}"])

        print(f"{label:36s} {means['rouge1']:8.4f} {means['rouge2']:8.4f} "
              f"{means['rougeLsum']:8.4f}")

    print(
        "\nNOTE: intersection-scored, not each run's own full split -- see "
        "intersection_report.json for what was dropped from each run to get "
        "here. The report separates each run's own infeasible/missing rows "
        "from feasible rows excluded only for pairing. No significance test "
        "is run by this script; per_example.jsonl "
        "in each run's output directory is the paired input a bootstrap or "
        "Wilcoxon test over this same intersection would use."
    )


if __name__ == "__main__":
    main()
