"""Where do the selected sentences come from?

Produces the three numbers used to diagnose F-0:
  * position distribution of selected sentences (0 = doc start, 1 = doc end)
  * overlap between the system's picks and Lead's picks
  * overlap between the system's picks and a greedy reference

A system whose picks look like Lead's, and rarely match the greedy reference, is
re-deriving Lead at much higher cost.

The greedy reference here is NOT an exact upper bound and NOT any dataset's
official oracle protocol -- it is a reproducible internal reference.

Historical diagnostic only. New governed runs must first filter the frozen
validation dev/dev-test manifest; test access is prohibited before freeze.
"""
from __future__ import annotations

import argparse
import statistics
from typing import List, Set

from src.eval.rouge import _new_scorer
from src.eval.oracle import greedy_reference_summary
from scripts.audit.lead_vs_system import load_jsonl, norm_sentences, reference_of


def summarise(name: str, positions: List[float]) -> None:
    if not positions:
        print(f"  {name:34s} (no data)")
        return
    front = sum(1 for p in positions if p <= 0.25) / len(positions)
    print(f"  {name:34s} median = {statistics.median(positions):.3f}   "
          f"front25% = {front:6.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--budget", type=int, default=None, help="word budget")
    ap.add_argument("--max_sentences", type=int, default=None)
    ap.add_argument("--limit", type=int, default=200,
                    help="docs to analyse (greedy reference is expensive)")
    args = ap.parse_args()

    raw = load_jsonl(args.data, args.limit)
    docs = {str(d["id"]): d for d in raw}
    preds = {str(r["id"]): r for r in load_jsonl(args.pred)}
    sc = _new_scorer(("rouge1",), True)

    sys_pos: List[float] = []
    ref_pos: List[float] = []
    lead_pos: List[float] = []
    ov_lead: List[float] = []
    ov_ref: List[float] = []
    n_used = 0

    for key in [str(d["id"]) for d in raw]:
        if key not in preds:
            continue
        doc = docs[key]
        sents = norm_sentences(doc)
        n = len(sents)
        if n < 2:
            continue
        S: Set[int] = set(preds[key].get("selected_indices", []))
        if not S:
            continue
        O: Set[int] = set(
            greedy_reference_summary(
                sents,
                reference_of(doc),
                max_words=args.budget,
                max_sentences=args.max_sentences,
                metric="rouge1",
                scorer=sc,
            )
        )
        L: Set[int] = set()
        tot = 0
        for i, s in enumerate(sents):
            if args.max_sentences is not None and len(L) >= args.max_sentences:
                break
            w = len(s.split())
            if args.budget is not None and tot + w > args.budget:
                break
            L.add(i)
            tot += w

        sys_pos += [i / (n - 1) for i in S]
        ref_pos += [i / (n - 1) for i in O]
        lead_pos += [i / (n - 1) for i in L]
        ov_lead.append(len(S & L) / len(S))
        ov_ref.append(len(S & O) / len(S))
        n_used += 1

    print(f"docs analysed = {n_used}\n")
    print("Position of selected sentences (0 = start of document, 1 = end):")
    summarise("Greedy reference (target)", ref_pos)
    summarise("Lead (trivial baseline)", lead_pos)
    summarise("System run", sys_pos)
    print()
    print(f"  System picks also chosen by Lead              : "
          f"{statistics.mean(ov_lead):6.1%}")
    print(f"  System picks also chosen by greedy reference  : "
          f"{statistics.mean(ov_ref):6.1%}")
    print()
    print("Reading: if the system's position profile sits near Lead's rather "
          "than the reference's, it is behaving like an expensive Lead.")


if __name__ == "__main__":
    main()
