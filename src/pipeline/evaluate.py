import argparse
import csv
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.data.schemas import extract_references
from src.eval.feasibility import classify_feasibility_row
from src.eval.protocol import KNOWN_PROTOCOLS, evaluate_corpus
from src.utils.io import read_jsonl


def align_evaluation_rows(
    prediction_rows: Iterable[Dict[str, Any]],
    gold_rows: Iterable[Dict[str, Any]],
    *,
    include_ids: Optional[Set[str]] = None,
) -> Tuple[List[str], List[List[str]]]:
    """Join predictions to gold references by ID and reject partial alignment.

    ``include_ids``, when given, limits which rows are actually scored while
    every row still counts toward the completeness check below -- a row
    excluded from scoring (e.g. marked infeasible) is not the same as a row
    missing from the artifact, and must not trip the ID-mismatch guard.
    """

    gold_by_id: Dict[str, List[str]] = {}
    for line_number, row in enumerate(gold_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"gold row {line_number} has no valid 'id'")
        if row_id in gold_by_id:
            raise ValueError(f"duplicate gold id: {row_id}")
        references = extract_references(row)
        if not references:
            raise ValueError(f"gold row {line_number} has no non-empty references")
        gold_by_id[row_id] = references

    predictions: List[str] = []
    references: List[List[str]] = []
    prediction_ids = set()
    for line_number, row in enumerate(prediction_rows, start=1):
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"prediction row {line_number} has no valid 'id'")
        if row_id in prediction_ids:
            raise ValueError(f"duplicate prediction id: {row_id}")
        prediction_ids.add(row_id)
        if not isinstance(row.get("summary"), str):
            raise ValueError(f"prediction row {line_number} must contain a string 'summary'")
        if row_id not in gold_by_id:
            raise ValueError(f"prediction id {row_id!r} is missing from the gold dataset")
        if include_ids is not None and row_id not in include_ids:
            continue
        predictions.append(row["summary"])
        references.append(gold_by_id[row_id])

    missing_predictions = set(gold_by_id) - prediction_ids
    if missing_predictions:
        preview = sorted(missing_predictions)[:5]
        raise ValueError(
            f"gold/prediction ID mismatch: {len(missing_predictions)} gold IDs have no "
            f"prediction; first IDs: {preview}"
        )
    return predictions, references


def plan_feasibility_scoring(
    prediction_rows: List[Dict[str, Any]],
    *,
    feasible_only: bool,
    assume_legacy_feasible: bool,
) -> Tuple[Optional[Set[str]], Dict[str, Any]]:
    """Classify prediction rows by feasibility and decide which IDs to score.

    Returns ``(include_ids, stats)``. ``include_ids`` is ``None`` when every
    row should be scored regardless of feasibility (``feasible_only=False``);
    otherwise it is the set of row IDs that are feasible, or assumed feasible
    under an explicitly acknowledged legacy (pre-F-17) schema.
    """

    total = 0
    feasible_ids: Set[str] = set()
    feasible_count = 0
    infeasible_count = 0
    legacy_count = 0
    schema_count = sum("feasible" in row for row in prediction_rows)
    if 0 < schema_count < len(prediction_rows):
        raise ValueError(
            "prediction artifact mixes F-17 rows with legacy rows missing "
            "'feasible'; rerun or migrate the artifact as one schema version"
        )
    for row in prediction_rows:
        total += 1
        row_id = row.get("id")
        feasible, used_legacy_assumption = classify_feasibility_row(
            row, assume_legacy_feasible=assume_legacy_feasible
        )
        if used_legacy_assumption:
            legacy_count += 1
        if feasible:
            feasible_count += 1
            feasible_ids.add(row_id)
        else:
            infeasible_count += 1

    stats = {
        "total_rows": total,
        "feasible_rows": feasible_count,
        "infeasible_rows": infeasible_count,
        "legacy_schema_assumed_feasible_rows": legacy_count,
        "scoring_mode": "feasible_only" if feasible_only else "all_rows",
    }
    include_ids = feasible_ids if feasible_only else None
    return include_ids, stats


def load_evaluation_inputs(
    prediction_path: str,
    gold_path: str,
    *,
    feasible_only: bool = False,
    assume_legacy_feasible: bool = False,
) -> Tuple[List[str], List[List[str]], Dict[str, Any]]:
    raw_predictions = list(read_jsonl(prediction_path))
    include_ids, stats = plan_feasibility_scoring(
        raw_predictions,
        feasible_only=feasible_only,
        assume_legacy_feasible=assume_legacy_feasible,
    )
    if feasible_only and not include_ids:
        raise ValueError(
            "feasible-only evaluation requested, but the artifact contains zero "
            "feasible rows; use the primary all-rows view and report feasibility"
        )
    predictions, references = align_evaluation_rows(
        raw_predictions, read_jsonl(gold_path), include_ids=include_ids
    )
    return predictions, references, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="predictions.jsonl path")
    ap.add_argument("--gold", required=True, help="canonical or legacy gold dataset JSONL")
    ap.add_argument("--out", required=True, help="metrics.csv output path")
    ap.add_argument(
        "--protocol",
        required=True,
        choices=KNOWN_PROTOCOLS,
        help="explicit dataset evaluation protocol",
    )
    ap.add_argument(
        "--feasible-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "diagnostic sensitivity: score only rows with feasible=true. "
            "The default primary view scores every input row so methods share "
            "the dataset denominator; use paired_run_intersection.py for a "
            "common-feasible comparison across methods."
        ),
    )
    ap.add_argument(
        "--assume-legacy-feasible",
        action="store_true",
        help=(
            "required to evaluate an artifact predating the F-17 'feasible' "
            "field; assumes every such row is feasible and records that "
            "assumption in the output metrics rather than silently defaulting to it. "
            "This schema opt-in does not waive full gold-ID coverage; use the "
            "paired intersection audit for explicitly diagnostic partial artifacts."
        ),
    )
    args = ap.parse_args()

    preds, refs, feasibility_stats = load_evaluation_inputs(
        args.pred,
        args.gold,
        feasible_only=args.feasible_only,
        assume_legacy_feasible=args.assume_legacy_feasible,
    )

    if feasibility_stats["legacy_schema_assumed_feasible_rows"]:
        print(
            "WARNING: assumed "
            f"{feasibility_stats['legacy_schema_assumed_feasible_rows']} "
            "legacy-schema row(s) (no top-level 'feasible' field) are feasible "
            "via --assume-legacy-feasible; this is an unverified assumption and "
            "is recorded in the output metrics."
        )
    print(
        f"{feasibility_stats['feasible_rows']}/{feasibility_stats['total_rows']} "
        f"rows feasible ({feasibility_stats['scoring_mode']}); "
        f"{feasibility_stats['infeasible_rows']} excluded from scoring"
        if args.feasible_only
        else
        f"{feasibility_stats['feasible_rows']}/{feasibility_stats['total_rows']} "
        f"rows feasible ({feasibility_stats['scoring_mode']}, all rows scored "
        "regardless of feasibility)"
    )

    t0 = time.perf_counter()
    m = evaluate_corpus(preds, refs, protocol=args.protocol)
    t1 = time.perf_counter()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["evaluation_protocol", args.protocol])
        # write rouge metrics
        for k, v in m.items():
            w.writerow([k, f"{v:.6f}"])
        # feasibility scoring metadata -- an assumption about unverifiable
        # legacy rows must be a recorded fact in the artifact, not a silent default
        w.writerow(["feasibility_scoring_mode", feasibility_stats["scoring_mode"]])
        w.writerow(["feasibility_total_rows", feasibility_stats["total_rows"]])
        w.writerow(["feasibility_feasible_rows", feasibility_stats["feasible_rows"]])
        w.writerow(["feasibility_infeasible_rows", feasibility_stats["infeasible_rows"]])
        w.writerow(
            [
                "feasibility_legacy_schema_assumed_feasible_rows",
                feasibility_stats["legacy_schema_assumed_feasible_rows"],
            ]
        )
        # append time statistics
        # selection time (if produced by select_sentences in the same directory)
        sel_time_file = os.path.join(os.path.dirname(args.out), "time_select_seconds.txt")
        if os.path.exists(sel_time_file):
            with open(sel_time_file, "r", encoding="utf-8") as fr:
                val = float((fr.read() or "0").strip())
                w.writerow(["time_select_seconds", f"{val:.6f}"])
        # evaluation time
        w.writerow(["time_eval_seconds", f"{(t1 - t0):.6f}"])
    print(f"ROUGE written to {args.out}")


if __name__ == "__main__":
    main()

