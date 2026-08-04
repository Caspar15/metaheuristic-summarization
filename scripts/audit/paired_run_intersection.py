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

Every run's *existing* rows in a pre-F-17 artifact are feasible by
construction: the documents that failed were never written at all, so
whatever is on disk already only contains successes. That is exactly the
condition ``--assume-legacy-feasible`` in ``src.pipeline.evaluate`` is meant
to certify, and this script requires the same explicit flag for the same
reason -- absent rows stay absent either way; this only avoids re-asserting
feasibility for the ones already recorded as present.

USAGE
-----
    python -m scripts.audit.paired_run_intersection \\
      --pred runs/greedy_mean/predictions.jsonl \\
             runs/greedy_length_normalized/predictions.jsonl \\
             runs/nsga2_mean/predictions.jsonl \\
      --gold data/processed/multi_news_validation_canonical.jsonl \\
      --out_dir runs/audit/f18_intersection_recompute \\
      --assume-legacy-feasible

Writes, under --out_dir:
    include_ids.json          sorted list of IDs scored (the intersection)
    intersection_report.json  per-run total/feasible/excluded-from-intersection counts
    <label>/per_example.jsonl one row per document: id, rouge1, rouge2, rougeLsum
    <label>/metrics.csv       corpus-mean metrics over the intersection only

<label> defaults to the basename of each --pred file's parent directory;
override with --labels if two runs share a parent directory name.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Set

from src.data.schemas import extract_references
from src.eval.rouge import rouge_scores
from src.pipeline.evaluate import plan_feasibility_scoring
from src.utils.io import read_jsonl


def _default_label(pred_path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.abspath(pred_path))) or pred_path


def _load_run(pred_path: str, *, assume_legacy_feasible: bool) -> Dict:
    rows = list(read_jsonl(pred_path))
    include_ids, stats = plan_feasibility_scoring(
        rows, feasible_only=True, assume_legacy_feasible=assume_legacy_feasible
    )
    pred_by_id = {row["id"]: row["summary"] for row in rows}
    return {"path": pred_path, "pred_by_id": pred_by_id,
            "include_ids": include_ids, "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pred", nargs="+", required=True,
                    help="two or more predictions.jsonl paths to compare")
    ap.add_argument("--gold", required=True, help="canonical gold dataset jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--labels", nargs="+", default=None,
                     help="one label per --pred, same order (default: parent dir name)")
    ap.add_argument(
        "--assume-legacy-feasible",
        action="store_true",
        help=(
            "required for a pre-F-17 artifact with no top-level 'feasible' "
            "field; every row present in such an artifact is feasible by "
            "construction (infeasible documents were never written), so this "
            "certifies that fact rather than guessing it."
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
    for row in read_jsonl(args.gold):
        gold_by_id[row["id"]] = extract_references(row)
    missing_from_gold = [i for i in sorted_ids if i not in gold_by_id]
    if missing_from_gold:
        raise ValueError(
            f"{len(missing_from_gold)} intersection ID(s) are not in --gold; "
            f"first: {missing_from_gold[:5]}"
        )
    references = [gold_by_id[i] for i in sorted_ids]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "include_ids.json"), "w", encoding="utf-8") as f:
        json.dump(sorted_ids, f, ensure_ascii=False, indent=2)

    intersection_report = {
        "intersection_size": len(sorted_ids),
        "runs": [
            {
                "label": label,
                "path": run["path"],
                "total_rows": run["stats"]["total_rows"],
                "feasible_rows": run["stats"]["feasible_rows"],
                "legacy_schema_assumed_feasible_rows": run["stats"][
                    "legacy_schema_assumed_feasible_rows"
                ],
                "excluded_from_intersection": sorted(
                    run["include_ids"] - common_ids
                ),
            }
            for label, run in zip(labels, runs)
        ],
    }
    with open(
        os.path.join(args.out_dir, "intersection_report.json"), "w", encoding="utf-8"
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
        with open(per_example_path, "w", encoding="utf-8") as f:
            for doc_id, scores in zip(sorted_ids, per_example):
                f.write(json.dumps({"id": doc_id, **scores}, ensure_ascii=False) + "\n")

        metrics_path = os.path.join(run_dir, "metrics.csv")
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["intersection_size", len(sorted_ids)])
            for metric, value in means.items():
                w.writerow([metric, f"{value:.6f}"])

        print(f"{label:36s} {means['rouge1']:8.4f} {means['rouge2']:8.4f} "
              f"{means['rougeLsum']:8.4f}")

    print(
        "\nNOTE: intersection-scored, not each run's own full split -- see "
        "intersection_report.json for what was dropped from each run to get "
        "here. No significance test is run by this script; per_example.jsonl "
        "in each run's output directory is the paired input a bootstrap or "
        "Wilcoxon test over this same intersection would use."
    )


if __name__ == "__main__":
    main()
