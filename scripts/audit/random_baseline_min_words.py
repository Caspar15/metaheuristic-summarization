"""Empirical support for the Random baseline's ``apply_min_words`` decision.

``src/baselines/random_baseline.py`` used to argue for ``apply_min_words=True``
based on a 400-row sample. PR #11 review flagged that sample as too small: it
found ``validation_4576`` (``source_capacity_words=244``, ``min_words_relaxed=
False``) failing under the skip-tolerant selector for seeds 0 and 42, which
contradicted the "zero failures across all four seeds and all 400 rows" claim
the old docstring made. This script reruns both selectors -- the naive
stop-at-first-miss ordering and the skip-tolerant walk this project actually
ships -- against the **full** validation split (5,621 rows) so the docstring
can cite a number that was not itself picked on a subsample.

Also computes the pool-vs-selected average sentence length used in
``random_baseline.py``'s "naming honesty" section: skipping sentences that
currently do not fit systematically favors shorter sentences, so "Random" is
better described as random-order greedy packing than uniform sampling.

Usage
-----
    python -m scripts.audit.random_baseline_min_words \
        --data data/processed/multi_news_validation_canonical.jsonl \
        --max_words 250 --min_words 200 --seeds 0,1,42,9999
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List

from src.baselines.contract import derive_row_seed, summarize_one_baseline
from src.baselines.random_baseline import _select_random
from src.objectives.evaluator import SelectionObjective


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _select_stop_at_first_miss(
    eligible_records: List[Dict[str, Any]],
    evaluator: SelectionObjective,
    rng: random.Random,
) -> List[int]:
    """Shuffle order, then stop (not skip) at the first sentence that does
    not currently fit -- structurally identical to Lead's own stopping rule,
    just over a random permutation instead of reading order."""

    order = list(range(len(eligible_records)))
    rng.shuffle(order)
    selected: List[int] = []
    for relative_index in order:
        if not evaluator.can_add(selected, relative_index):
            break
        selected.append(relative_index)
    return selected


def _run_variant(
    rows: List[Dict],
    select_fn,
    seeds: List[int],
    max_words: int,
    min_words: int,
) -> Dict[int, List[str]]:
    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": max_words,
            "min_words": min_words,
            "require_nonempty": True,
        }
    }
    failures: Dict[int, List[str]] = {}
    for seed in seeds:
        seed_failures = []
        for doc in rows:
            try:
                summarize_one_baseline(
                    doc,
                    cfg,
                    method="diagnostic",
                    select_fn=select_fn,
                    length_gate=True,
                    apply_min_words=True,
                    seed=seed,
                    requires_seed=True,
                )
            except ValueError:
                seed_failures.append(doc.get("id"))
        failures[seed] = seed_failures
    return failures


def _pool_word_lengths(rows: List[Dict]):
    """Seed-independent: computed once, not once per seed."""
    from src.data.schemas import flatten_sentence_records
    from src.utils.tokenizer import count_tokens

    pool_lengths = []
    for doc in rows:
        sentence_records = flatten_sentence_records(doc)
        pool_lengths.extend(count_tokens(r["text"]) for r in sentence_records)
    return pool_lengths


def _selected_word_stats(rows: List[Dict], seed: int, max_words: int, min_words: int):
    from src.utils.tokenizer import count_tokens

    cfg = {
        "length_control": {
            "unit": "words",
            "max_words": max_words,
            "min_words": min_words,
            "require_nonempty": True,
        }
    }
    selected_lengths = []
    selected_counts = []
    skipped = 0
    for doc in rows:
        try:
            result = summarize_one_baseline(
                doc,
                cfg,
                method="diagnostic",
                select_fn=_select_random,
                length_gate=True,
                apply_min_words=True,
                seed=seed,
                requires_seed=True,
            )
        except ValueError:
            # A handful of rows are infeasible for this selector/seed (the
            # same rows _run_variant reports as failures above); excluded
            # from the word-length stats since there is no selection to
            # measure, not silently averaged in as zero.
            skipped += 1
            continue
        selected_lengths.extend(count_tokens(s) for s in result["summary_sentences"])
        selected_counts.append(len(result["summary_sentences"]))
    if skipped:
        print(f"  (excluded {skipped} infeasible rows from word-length stats)")
    return selected_lengths, selected_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max_words", type=int, default=250)
    ap.add_argument("--min_words", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,42,9999")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    rows = load_jsonl(args.data)
    n = len(rows)
    print(f"Loaded {n} rows from {args.data}")

    for label, select_fn in (
        ("stop-at-first-miss (naive, Lead-shaped)", _select_stop_at_first_miss),
        ("skip-tolerant (this module's _select_random)", _select_random),
    ):
        failures = _run_variant(rows, select_fn, seeds, args.max_words, args.min_words)
        print(f"\n{label}:")
        for seed in seeds:
            pct = 100.0 * len(failures[seed]) / n
            print(f"  seed={seed}: {len(failures[seed])}/{n} failed ({pct:.2f}%)")
            if failures[seed]:
                print(f"    example ids: {failures[seed][:5]}")

    print("\nPool vs. selected sentence word-length (naming honesty), per seed:")
    pool_lengths = _pool_word_lengths(rows)
    for seed in seeds:
        selected_lengths, selected_counts = _selected_word_stats(
            rows, seed=seed, max_words=args.max_words, min_words=args.min_words
        )
        print(
            f"  seed={seed}: selected avg words/sentence="
            f"{sum(selected_lengths) / len(selected_lengths):.2f} (n={len(selected_lengths)}), "
            f"avg selected sentence count/doc={sum(selected_counts) / len(selected_counts):.2f}"
        )
    print(f"  pool avg words/sentence: {sum(pool_lengths) / len(pool_lengths):.2f} (n={len(pool_lengths)})")


if __name__ == "__main__":
    main()
