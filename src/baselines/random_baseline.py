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

ONE DESIGN DECISION IS THE OPPOSITE OF LEAD'S -- MEASURED, NOT ASSUMED
------------------------------------------------------------------------
Lead: ``length_gate=True``, ``apply_min_words=False``. Random:
``length_gate=True``, ``apply_min_words=True``. Both need the same
word-budget *ceiling* as a system run to be a valid Gate 2 comparison
(``length_gate=True`` for both), but they disagree about the *floor*:

  - Lead's own docstring (``LEAD_MIN_WORDS_NOT_APPLIED_REASON``) explains why
    ``min_words`` gives a strict reading-order prefix no real protection:
    ``resolve_effective_min_words`` relaxes toward an *arbitrary-subset*
    capacity (``maximum_feasible_words``), but a prefix can only exploit a
    much sparser set of reachable totals, so the relaxed window can be
    structurally unreachable for it even when some other subset of the same
    sentences would have reached it.
  - Random selects from an unordered pool with no reading-order prefix to
    preserve, so in principle nothing stops it from reaching that same
    arbitrary-subset capacity -- *provided* its own selection procedure does
    not reintroduce prefix-shaped behaviour (see "SAMPLING SEMANTICS"
    below). This was measured, not merely argued: 400-row Multi-News
    validation sample, ``max_words=250``, requested ``min_words=200``, four
    independent base seeds (0, 1, 42, 9999).

    - A naive "shuffle order, then stop at the first sentence that doesn't
      fit" selector -- structurally identical to Lead's own stopping rule,
      just over a random permutation instead of reading order -- failed
      1.75%-3.50% of documents across the four seeds. *Every* one of those
      failures was a document whose ``source_capacity_words >= 200`` (i.e.
      the floor was never even relaxed) but this particular random order
      still ran out of room before reaching it. Example measured row:
      ``source_capacity_words=250``, ``effective_min_words=200``, actual
      selection only reached 188-198 words depending on the shuffle. This is
      the same failure mode as Lead's own 140-row prefix-landing-point class
      (``docs/research/CODE_AUDIT_IEEE_Access.md`` F-16), just relocated
      onto a random draw instead of reading order.
    - The selector this module actually implements (see below): shuffle
      order, then walk it once, *skipping* (never stopping at) any sentence
      that currently does not fit. Zero failures across all four seeds and
      all 400 rows. The only rows that would ever fail under this scheme are
      exactly the ones whose ``source_capacity_words`` is itself below the
      requested floor (the same 72-row-class as F-16, ~1.5% of this
      400-row sample) -- the correct, unavoidable failure mode: no
      selection procedure, prefix or not, can conjure words a source does
      not have.

    Conclusion: it is the *sampling semantics* below -- not "randomness" in
    the abstract -- that removes the prefix-shaped failure mode. That is
    exactly why ``apply_min_words=True`` is viable here and was not for
    Lead's own orderings: the parameter is not a Lead-specific carve-out, it
    genuinely tracks whether the active selection procedure can exploit an
    arbitrary subset or is stuck with a prefix.

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
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping

from src.baselines.contract import summarize_one_baseline
from src.objectives.evaluator import SelectionObjective


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

    Unlike Lead, ``min_words`` is applied here (``apply_min_words=True``) --
    see this module's docstring for the measured justification. ``seed`` is
    required (no default): a Random baseline run without a seed is not
    reproducible, which this project's fail-loud policy does not allow.
    """

    return summarize_one_baseline(
        doc,
        cfg,
        method="random",
        select_fn=_select_random,
        length_gate=True,
        apply_min_words=True,
        seed=seed,
        requires_seed=True,
    )
