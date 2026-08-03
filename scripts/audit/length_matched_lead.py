"""Build the length bracket around a system run's output length.

WHY THIS EXISTS
---------------
A system that spends more words than Lead can win ROUGE-1 / ROUGE-Lsum
simply by saying more. The ICT Express audit found exactly that pattern in
the legacy run (system used ~13 more whitespace words than Lead), and it is
the reason a bare "system vs Lead at a fixed budget" comparison cannot
settle F-0.

Sentence granularity makes an exact per-document length match impossible: a
reading-order prefix either stops before the target (undershoot) or steps
past it (overshoot). Reporting only one of the two is what makes a length
argument unfalsifiable, so this script emits BOTH:

    Lead(undershoot)  <=  system words  <=  Lead(overshoot)

If the system beats both, the win is not a length artifact. If it only beats
the undershoot variant, it is.

USAGE
-----
    python -m scripts.audit.length_matched_lead \\
      --data data/processed/multi_news_validation_canonical.jsonl \\
      --pred <run>/predictions.jsonl \\
      --out_dir <run>/length_bracket

Then score each emitted file with ``src.pipeline.evaluate`` against a gold
file restricted to the same ids (``--pred`` may cover fewer rows than
``--data`` if the selector could not make some documents feasible; see F-17).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Dict, List

from src.data.schemas import flatten_sentence_records
from src.utils.io import read_jsonl
from src.utils.tokenizer import count_tokens


def system_word_counts(pred_path: str) -> Dict[str, int]:
    """Per-document output length of the system run, in whitespace words.

    ``src.pipeline.select_sentences`` rows carry no
    ``output_budget.selected_words`` -- that field is written by the baseline
    contract only -- so fall back to counting the emitted summary. Both paths
    count the same way (``str.split()``), so the two are interchangeable.
    """

    out: Dict[str, int] = {}
    for row in read_jsonl(pred_path):
        budget = row.get("output_budget") or {}
        words = budget.get("selected_words")
        if words is None:
            words = len(" ".join(row.get("summary_sentences", [])).split())
        out[row["id"]] = int(words)
    return out


def _lead_row(row, records, picked, total: int, target: int) -> dict:
    return {
        "id": row["id"],
        "selected_indices": list(range(len(picked))),
        "selected_sentences": picked,
        "summary_sentences": [r["text"] for r in picked],
        "summary": "\n".join(r["text"] for r in picked),
        "output_budget": {"selected_words": total, "matched_to_system": target},
    }


def build(data_path: str, targets: Dict[str, int], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    under_lens: List[int] = []
    over_lens: List[int] = []
    ran_out = 0

    under_path = os.path.join(out_dir, "lead_undershoot.jsonl")
    over_path = os.path.join(out_dir, "lead_overshoot.jsonl")

    with open(under_path, "w", encoding="utf-8") as f_under, \
            open(over_path, "w", encoding="utf-8") as f_over:
        for row in read_jsonl(data_path):
            if row["id"] not in targets:
                continue
            target = targets[row["id"]]
            records = flatten_sentence_records(row)

            # undershoot: never exceed the target (always keep >=1 sentence)
            picked, total = [], 0
            for rec in records:
                words = count_tokens(rec["text"])
                if picked and total + words > target:
                    break
                picked.append(rec)
                total += words
            under_lens.append(total)
            f_under.write(json.dumps(_lead_row(row, records, picked, total, target),
                                     ensure_ascii=False) + "\n")

            # overshoot: stop as soon as the target is reached or passed
            picked, total = [], 0
            for rec in records:
                picked.append(rec)
                total += count_tokens(rec["text"])
                if total >= target:
                    break
            if total < target:
                ran_out += 1
            over_lens.append(total)
            f_over.write(json.dumps(_lead_row(row, records, picked, total, target),
                                    ensure_ascii=False) + "\n")

    sys_words = [targets[i] for i in sorted(targets)]
    print(f"rows                  : {len(under_lens)}")
    print(f"system mean words     : {statistics.mean(sys_words):.1f}")
    print(f"Lead undershoot mean  : {statistics.mean(under_lens):.1f} "
          f"({statistics.mean(under_lens) - statistics.mean(sys_words):+.1f})")
    print(f"Lead overshoot  mean  : {statistics.mean(over_lens):.1f} "
          f"({statistics.mean(over_lens) - statistics.mean(sys_words):+.1f})")
    if ran_out:
        print(f"documents whose source ran out before reaching the target: {ran_out}")
    print(f"wrote {under_path}")
    print(f"wrote {over_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="canonical dataset jsonl")
    ap.add_argument("--pred", required=True, help="system predictions.jsonl")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    targets = system_word_counts(args.pred)
    if not targets:
        raise ValueError(f"no prediction rows found in {args.pred}")
    build(args.data, targets, args.out_dir)


if __name__ == "__main__":
    main()
