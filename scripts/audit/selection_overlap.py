"""How much do two runs pick the same sentences?

WHY THIS EXISTS
---------------
The architecture rewrite was justified by a legacy diagnostic showing the old
system selected 61.7% of the same sentences Lead would select
(``selection_diagnostics.py``, 200-document sample of the old test split, on
a test-tuned artifact). That number is the reason the candidate funnel was
rewired, so "did the overlap actually fall?" is the leading indicator the
validation pilot has to check before spending compute on a full run
(``ACTION_PLAN.md`` Phase 3).

This script measures the same quantity properly: full split, both sides
produced by the current pipeline, matched by ``sentence_id`` rather than
position (so it is unaffected by output ordering).

⚠️ The 61.7% is NOT a comparable baseline for the output of this script --
different split, different sample size, different evaluator, test-tuned
artifact. Direction is meaningful; a subtraction is not.

USAGE
-----
    python -m scripts.audit.selection_overlap \\
      --a <system>/predictions.jsonl \\
      --b runs/gate2_lead_document_order_val/predictions.jsonl
"""
from __future__ import annotations

import argparse
import statistics
from typing import Dict, Set

from src.utils.io import read_jsonl


def selected_ids(path: str) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for row in read_jsonl(path):
        out[row["id"]] = {
            sentence["sentence_id"] for sentence in row.get("selected_sentences", [])
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run A predictions.jsonl (the system)")
    ap.add_argument("--b", required=True, help="run B predictions.jsonl (usually Lead)")
    args = ap.parse_args()

    a, b = selected_ids(args.a), selected_ids(args.b)
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("the two runs share no document ids")
    if len(common) != len(a) or len(common) != len(b):
        print(f"WARNING: row sets differ (A={len(a)} B={len(b)} "
              f"common={len(common)}); computed on the intersection")

    per_doc, jaccard = [], []
    a_sizes, b_sizes = [], []
    identical = 0
    for doc_id in common:
        sa, sb = a[doc_id], b[doc_id]
        a_sizes.append(len(sa))
        b_sizes.append(len(sb))
        if not sa:
            continue
        inter = len(sa & sb)
        per_doc.append(inter / len(sa))
        union = len(sa | sb)
        jaccard.append(inter / union if union else 0.0)
        if sa == sb:
            identical += 1

    print(f"documents compared            : {len(common)}")
    print(f"overlap |A n B| / |A|  (mean) : {100 * statistics.mean(per_doc):.1f}%")
    print(f"                     (median) : {100 * statistics.median(per_doc):.1f}%")
    print(f"Jaccard                (mean) : {100 * statistics.mean(jaccard):.1f}%")
    print(f"identical selections          : {identical} "
          f"({100 * identical / len(common):.1f}%)")
    print(f"sentences per summary         : A={statistics.mean(a_sizes):.2f}  "
          f"B={statistics.mean(b_sizes):.2f}")


if __name__ == "__main__":
    main()
