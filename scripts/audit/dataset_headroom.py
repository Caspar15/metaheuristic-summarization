"""How much room is there ABOVE Lead on each dataset?

headroom = greedy_reference_score - lead_score

If headroom is small, Lead is already near-optimal and no selector can win --
the dataset is the wrong battlefield. If greedy-reference sentences cluster at
the front of the document, the same conclusion follows.

The greedy reference is NOT an exact upper bound and NOT an official protocol;
it is a reproducible internal diagnostic used to compare datasets.

Usage
-----
    Historical diagnostic only. New governed runs must first filter the frozen
    validation dev/dev-test manifest; test access is prohibited before freeze.
"""
from __future__ import annotations

import argparse
import statistics

from src.eval.rouge import rouge_scores, _new_scorer
from src.eval.oracle import greedy_reference_summary
from scripts.audit.lead_vs_system import (
    load_jsonl, norm_sentences, reference_of, lead_prefix,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--budget", type=int, default=None, help="word budget")
    ap.add_argument("--lead_sentences", type=int, default=None)
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    docs = [d for d in load_jsonl(args.data, args.limit) if norm_sentences(d)]
    sc = _new_scorer(("rouge1",), True)

    refs, leads, orc, positions, nsent = [], [], [], [], []
    for d in docs:
        sents = norm_sentences(d)
        ref = reference_of(d)
        refs.append(ref)
        leads.append(lead_prefix(sents, max_words=args.budget,
                                 max_sentences=args.lead_sentences))
        idx = greedy_reference_summary(
            sents,
            ref,
            max_words=args.budget,
            max_sentences=args.lead_sentences,
            metric="rouge1",
            scorer=sc,
        )
        orc.append(" ".join(sents[i] for i in idx))
        n = len(sents)
        nsent.append(n)
        positions += [i / max(1, n - 1) for i in idx]

    L = rouge_scores(leads, refs, metrics=("rouge1",))["rouge1"]
    O = rouge_scores(orc, refs, metrics=("rouge1",))["rouge1"]
    med = statistics.median(positions) if positions else float("nan")
    front = sum(1 for p in positions if p <= 0.25) / max(1, len(positions))

    print(f"dataset            : {args.data}")
    print(f"docs               : {len(docs)}   median sentences/doc = {statistics.median(nsent):.0f}")
    print(f"Lead R-1           : {L:.4f}")
    print(f"Greedy ref R-1     : {O:.4f}")
    print(f"Headroom           : {O - L:.4f}")
    print(f"Reference position : median {med:.2f}   front25% {front:.1%}")
    print()
    print("High front25% means strong lead bias -- Lead is hard to beat there.")


if __name__ == "__main__":
    main()
