"""Random baseline for extractive summarization.

WHY THIS EXISTS
---------------
Phase 2's second baseline (docs/research/ACTION_PLAN.md, "Baseline 與 reality
check"), for two purposes: (1) a lower-bound sanity check -- any method that
does not beat random sentence sampling under the same word budget has a
problem well upstream of tuning or search strategy; (2) a stress test of the
abstraction ``src/baselines/contract.py`` was built for Lead -- does a
*second*, structurally different baseline reuse it cleanly, or was Lead's
contract actually Lead-shaped rather than baseline-shaped? The answer: not
quite as-is. Lead's ``SelectFn`` (records, evaluator) -> indices had nowhere
to carry a reproducible seed, so ``contract.py`` gained ``seed``/
``requires_seed``/``SeededSelectFn``/``derive_row_seed`` to close that gap
(see ``contract.py``'s own docstring for the full interface argument). This
module is the first concrete user of that addition.

MIN_WORDS DOES NOT APPLY HERE EITHER -- SAME GENERAL PRINCIPLE AS LEAD, NOT
A LEAD-SPECIFIC CARVE-OUT
------------------------------------------------------------------------
An earlier version of this module set ``apply_min_words=True`` for Random,
on the theory that Lead's exemption (``LEAD_MIN_WORDS_NOT_APPLIED_REASON`` in
``lead.py``) was about prefixes specifically, and Random -- drawing from an
unordered pool -- had no prefix to be stuck with. PR #11 review found that
theory did not survive a full-scale measurement (see below): it is reversed
here, to ``apply_min_words=False``, for a reason stated at the level that
actually governs it, not "Random also doesn't work":

  ``min_words`` is a constraint that a *searching* selector can satisfy by
  trying a different subset of sentences. This project's system selectors
  -- Greedy, GRASP, NSGA-II -- all search: if one candidate subset lands
  short of the floor, they can trade a sentence out for another and keep
  looking. A baseline does not search. It walks through its own selection
  rule -- a reading-order prefix for Lead, a shuffle-then-walk for Random --
  exactly once, and stops when that rule is done, whatever the outcome.
  ``resolve_effective_min_words``'s relaxation target, ``maximum_feasible_
  words``, is the *exact* optimum over an arbitrary subset (an
  ``O(n * max_length)`` bitset subset-sum). A single walk over one ordering
  -- reading order for Lead, one random permutation for Random -- is not an
  optimal bin packer, and the gap between "the arbitrary-subset optimum"
  and "what one non-searching walk happens to reach" is exactly where a
  baseline's infeasible rows come from. This holds whether the walk stops
  at the first sentence that doesn't fit (Lead's rule, and the naive
  variant measured below) or skips misses and keeps going (the selector
  this module actually implements, also measured below) -- skipping lowers
  the failure rate, it does not zero it out, because neither is a bin
  packer. Because none of this argument is stated in terms of Random
  specifically, it is expected to apply identically to any future
  non-searching baseline this project adds (TextRank, LexRank, MMR): if a
  baseline's ``select_fn`` does not search over subsets to satisfy a
  feasibility constraint, ``min_words`` is out of scope for it, full stop.

  **Measured, on the full 5,621-row Multi-News validation split** (not a
  400-row subsample -- see below for why that mattered), ``max_words=250``,
  requested ``min_words=200``, base seeds 0, 1, 42, 9999
  (``scripts/audit/random_baseline_min_words.py``, run 2026-08-03 against
  ``data/processed/multi_news_validation_canonical.jsonl``):

  - Naive "shuffle order, then stop at the first sentence that doesn't fit"
    (structurally identical to Lead's own stopping rule, just over a random
    permutation instead of reading order): failed on 2.31%-2.60% of
    documents across the four seeds (130-146 of 5,621). *Every* failure was
    a document whose ``source_capacity_words >= 200`` (the floor was never
    even relaxed) but this particular random order ran out of room before
    reaching it -- the same failure mode as Lead's own prefix-landing-point
    class (``docs/research/CODE_AUDIT_IEEE_Access.md`` F-16), relocated onto
    a random draw instead of reading order.
  - The skip-tolerant selector this module actually implements (``shuffle,
    then walk once, skipping rather than stopping at, anything that
    currently does not fit -- see "SAMPLING SEMANTICS" below): failed on
    0.04%-0.07% of documents across the four seeds (2-4 of 5,621) -- far
    lower than the naive rule, but **not zero**. The earlier 400-row
    measurement had reported zero failures for this selector and concluded
    ``apply_min_words=True`` was therefore safe; that conclusion did not
    hold at full scale, it just had not been looked at hard enough to find
    the failures. ``validation_4576`` (``source_capacity_words=244``,
    ``min_words_relaxed=False``, 7 eligible sentences of lengths
    ``[45, 22, 72, 12, 22, 83, 7]``) fails under **all four** seeds tested,
    landing at 180-191 words: the 244-word optimum requires keeping *both*
    of the two long sentences (72 and 83 words) while dropping the two
    shortest, a specific combination a single random walk only rarely lands
    on. This is the concrete instance of "not an optimal bin packer" above,
    not a hypothetical.

  A rejected alternative, recorded here rather than silently dropped:
  retry the shuffle with a fresh permutation until one reaches the floor,
  instead of turning the constraint off. Rejected for two independent
  reasons: (1) it makes Random a *search* procedure -- exactly the
  distinction this section draws between baselines and system selectors,
  so a retrying Random would no longer be testing what a baseline is
  supposed to test; (2) accepting only permutations that happen to clear
  the floor is itself a biased sampling scheme -- it conditions the
  reported distribution of summaries on an outcome (feasibility) correlated
  with which sentences got selected, which is a different and undocumented
  sampling procedure, not the same "Random" with a retry loop bolted on.

SAMPLING SEMANTICS -- A SELF-DEFINED CHOICE, NOT A CITABLE CONVENTION
-----------------------------------------------------------------------
As with Lead's multi-document ordering, a literature check (2026-07) found
no single citable convention for what "Random baseline" means operationally:
descriptions found range from "the same sentence count as the reference,
chosen at random" to vaguer "uniform sampling" statements with no budget
specified, and neither settles sentence-count vs. word-budget. Fabbri et al.
2019 (the Multi-News paper itself) does not report a Random baseline at all,
so there is no dataset-specific precedent to match either. This module
therefore *defines its own*, for reasons specific to this project rather
than because the literature says so:

  - Sample under the same **word budget** as Lead/system runs, not a
    randomly chosen sentence count. Deciding a random ``k`` first and then
    drawing ``k`` sentences would decouple Random from the fixed-length
    comparison Gate 2 requires, and the literature does not settle this
    question either -- word-budget sampling is what this project needs to
    stay comparable, so that is what is implemented.
  - Shuffle eligible sentences, then walk the shuffled order once, adding
    any sentence that still fits and *skipping* (never stopping at) one
    that does not. This is deliberately not Lead's strict-prefix-stop rule:
    that rule exists for Lead to preserve a faithful reading-order prefix,
    and once selection order is random there is no ordering integrity left
    to protect by stopping early -- "stop at the first miss" would just
    reproduce Lead's own prefix-landing weakness for no reason (see the
    measured numbers above). Nothing citable settles this choice either; it
    is justified by the argument above, not by convention.
  - Output order: unaffected by either choice above.
    ``summarize_one_baseline`` already sorts the final selected indices back
    into original document position (``selected = sorted(...)``) regardless
    of what order ``select_fn`` picked them in, so Random's output is in
    reading order like every other baseline built on this contract, with no
    extra handling needed here. This also matches the general
    extractive-summarization convention of presenting output in source
    order regardless of the selection criterion.

NAMING HONESTY -- THIS IS NOT UNIFORM RANDOM SAMPLING
------------------------------------------------------
"Random" is a convenient short label, not an accurate description of what
``_select_random`` draws. Skipping (rather than stopping at) a sentence that
does not currently fit means a sentence's odds of being included depend on
how much budget is left when its turn in the shuffle comes up, which in turn
depends on its own length relative to every other sentence's length -- a
uniform draw over shuffles does not translate into a uniform draw over
which sentences end up selected. Concretely, this is a **random-order
greedy packing** (or "random-order first-fit"), and it measurably favors
shorter sentences: a full 5,621-row rerun
(``scripts/audit/random_baseline_min_words.py``, ``max_words=250``,
``min_words=200``, 2026-08-03) found the eligible pool averaging 21.55
words/sentence, versus 18.74-18.83 words/sentence among the ~13.1-13.2
sentences actually selected per document, across seeds 0/1/42/9999. A
longer sentence is simply less likely to fit into whatever budget happens
to remain by the time the shuffle reaches it. Any paper text describing this
baseline should say "random-order greedy packing" (or an equivalent
explicit phrase) rather than bare "Random", which would misrepresent it as
an unbiased draw.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping

from src.baselines.contract import summarize_one_baseline
from src.objectives.evaluator import SelectionObjective

RANDOM_MIN_WORDS_NOT_APPLIED_REASON = (
    "min_words is not enforced for Random; see the 'MIN_WORDS DOES NOT "
    "APPLY HERE EITHER' section of src/baselines/random_baseline.py's "
    "module docstring for the full argument. General principle, not a "
    "Random-specific carve-out: min_words is a constraint that a "
    "*searching* selector can satisfy by trying a different subset -- "
    "this project's Greedy/GRASP/NSGA-II selectors all search, so they can "
    "trade one sentence for another to clear the floor. A baseline does "
    "not search; it walks through its own selection rule exactly once and "
    "stops when that rule is done, whatever the outcome. "
    "maximum_feasible_words (resolve_effective_min_words's relaxation "
    "target) is the exact optimum over an arbitrary subset; a single "
    "first-fit walk over one random permutation is not an optimal bin "
    "packer, and that gap is where a baseline's infeasible rows come from "
    "-- expected to apply identically to any future non-searching "
    "baseline (TextRank, LexRank, MMR), not just Random. Measured: even "
    "this module's skip-tolerant selector fails on 2-4 of 5,621 "
    "validation rows per seed (0.04%-0.07%, seeds 0/1/42/9999), including "
    "validation_4576 (source_capacity_words=244, min_words_relaxed=False) "
    "failing under all four seeds tested -- see "
    "scripts/audit/random_baseline_min_words.py for the reproducible "
    "measurement."
)


def _select_random(
    eligible_records: List[Dict[str, Any]],
    evaluator: SelectionObjective,
    rng: random.Random,
) -> List[int]:
    """Shuffle, then walk once, skipping (not stopping at) anything that
    does not currently fit. See this module's docstring, "SAMPLING
    SEMANTICS", for why this is not Lead's strict-prefix-stop rule."""

    order = list(range(len(eligible_records)))
    rng.shuffle(order)
    selected: List[int] = []
    for relative_index in order:
        if evaluator.can_add(selected, relative_index):
            selected.append(relative_index)
    return selected


def summarize_one_random(
    doc: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    seed: int,
) -> Dict[str, Any]:
    """Random-baseline analogue of ``select_sentences.summarize_one``.

    ``min_words`` is not applied here (``apply_min_words=False``), same as
    Lead -- see this module's docstring, "MIN_WORDS DOES NOT APPLY HERE
    EITHER", for the general principle and the measured justification.
    ``seed`` is required (no default): a Random baseline run without a seed
    is not reproducible, which this project's fail-loud policy does not
    allow.
    """

    return summarize_one_baseline(
        doc,
        cfg,
        method="random",
        select_fn=_select_random,
        length_gate=True,
        apply_min_words=False,
        min_words_not_applied_reason=RANDOM_MIN_WORDS_NOT_APPLIED_REASON,
        seed=seed,
        requires_seed=True,
    )
