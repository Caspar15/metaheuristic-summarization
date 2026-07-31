"""Lead baseline for extractive summarization.

WHY THIS EXISTS
---------------
Phase 2 (docs/research/ACTION_PLAN.md, "Baseline 與 reality check") requires
Lead as the cheapest reality check before any metaheuristic method is taken
seriously: if the system does not beat Lead under an identical word budget
and an identical feasibility contract, nothing downstream matters. See
CLAUDE.md section 2, finding 1: the legacy Multi-News comparison already
failed this check once, in part because it was never actually run
length-matched.

MULTI-DOCUMENT ORDERING -- THIS IS A DELIBERATE, DOCUMENTED CHOICE, NOT A
LITERATURE CONVENTION
--------------------------------------------------------------------------
Multi-News clusters contain 2-8 source documents concatenated together, so
"Lead" is ambiguous in a way it is not for single-document CNN/DM Lead-3.
A literature check (2026-07) found:

  - Fabbri et al. 2019 (the original Multi-News paper, ACL 2019,
    arXiv:1906.01749), Section 6.1: "First-k means the first k sentences
    from each source article will be concatenated as the summary." This is
    a *per-document* sentence count (k sentences from *every* article, so
    total length scales with the number of source documents) -- it is NOT a
    fixed total word budget across the whole cluster, which is what this
    project's Gate 2 comparison actually needs (200-250 words regardless of
    cluster size). The two are not the same baseline and must not be
    conflated. The paper also does not specify the cross-document
    concatenation order, and the released code
    (github.com/Alex-Fabbri/Multi-News, code/extractive_code/) does not
    include a First/Lead script to check against either.
  - Follow-up papers checked (PRIMERA) do not define or report a Multi-News
    Lead/First-k baseline at all.

Conclusion: there is no citable convention for a fixed-word-budget Multi-News
Lead baseline. This module therefore *defines its own*, and that definition
must be stated explicitly in the paper -- never written up as "following the
Multi-News convention", because no such convention was found to exist for
this exact setting.

Three orderings are implemented so the choice is explicit rather than
implicit:

  - ``document_order`` (default, the one used for the formal Gate 2 number):
    walk sentences in source-cluster reading order (document 1 in full
    document/sentence order, then document 2, ...) and add each next
    eligible sentence while it still fits the active word/sentence budget;
    stop the first time the next sentence would not fit. This is a strict
    prefix in reading order, not a knapsack fill: a later, shorter sentence
    is never used to patch a gap left by an earlier one that didn't fit.
  - ``round_robin``: cycle across source documents (one sentence per
    document per round, in ascending document order) instead of exhausting
    one document before moving to the next. Same strict-prefix stopping
    rule, only the traversal order differs.
  - ``fabbri_first_k`` (diagnostic only): reproduces the original paper's
    own First-k definition -- the first ``first_k`` eligible sentences of
    *every* source document, concatenated in ascending document order (the
    paper does not specify this ordering choice either; this module states
    it explicitly). Because summary length scales with document count, this
    mode does not go through the fixed word-budget feasibility gate
    (``length_gate=False``) and its output must never be reported next to a
    length-matched Gate 2 comparison.

Every mode still goes through ``src.baselines.contract.summarize_one_baseline``,
which enforces the length upper bounds (``max_words``/``max_tokens``,
``max_sentences``, ``require_nonempty``) via ``src.objectives.evaluator``'s
shared ``resolve_selection_eligibility`` -- the exact same function a system
run uses -- rather than a re-derived approximation of it. Those upper bounds
still fail loud per document via ``assert_feasible`` rather than silently
padding or truncating the summary, consistent with this project's
no-silent-fallback rule (CLAUDE.md section 3). A document that fails this
way is a genuine fact about how hard Lead is on that document, not a bug to
paper over.

MIN_WORDS DOES NOT APPLY TO LEAD -- DECIDED AFTER AN EMPIRICAL CHECK, NOT
ASSUMED
--------------------------------------------------------------------------
The lower bound (``length_control.min_words``) is different from the upper
bounds above: every call into ``summarize_one_baseline`` below passes
``apply_min_words=False``, so ``SelectionConstraints.min_words`` is always 0
for Lead regardless of what the config requests. This is a deliberate
exception, not an oversight, for two independent reasons (see
``LEAD_MIN_WORDS_NOT_APPLIED_REASON`` below for the artifact-recorded form):

  1. Technical: ``resolve_effective_min_words`` relaxes a requested floor
     down to ``maximum_feasible_words``, which is the exact bitset
     subset-sum over an *arbitrary* subset of the eligible sentences. A
     strict reading-order prefix cannot exploit an arbitrary subset -- its
     attainable totals are a much sparser, discrete set of landing points --
     so the relaxed ``[min_words, max_words]`` window can be structurally
     unreachable for a prefix even when it is reachable for some other
     subset of the very same sentences. The relaxation therefore gives a
     prefix baseline no real protection against the failure mode it exists
     to prevent.
  2. Methodological: ``docs/research/ACTION_PLAN.md`` item 1e records why
     ``min_words=200`` exists at all -- as a guard against a mean-salience
     selection objective degenerating to a single high-scoring sentence.
     Lead has no selection objective to degenerate (it only ever fills the
     prefix greedily until the next sentence would no longer fit), so that
     guard does not apply to it either.

The empirical distribution of selected word counts under this exception
(the full-split min_words=0 measurement, zero infeasible documents, and the
cross-validated 72-vs-140-row breakdown of why 3.77% of documents land
under 200 words) is recorded in
``docs/research/CODE_AUDIT_IEEE_Access.md`` finding F-16, not duplicated
here -- a per-run number belongs in one place so it cannot go stale in two.
Every Lead row still records ``output_budget.min_words_applied: false``,
the untouched ``requested_min_words`` (never silently dropped), and
``LEAD_MIN_WORDS_NOT_APPLIED_REASON`` below, so the exception is auditable
per document rather than assumed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Tuple

from src.baselines.contract import SelectFn, summarize_one_baseline
from src.objectives.evaluator import SelectionObjective

ORDERINGS = ("document_order", "round_robin", "fabbri_first_k")

LEAD_MIN_WORDS_NOT_APPLIED_REASON = (
    "min_words is not enforced for Lead; see the 'MIN_WORDS DOES NOT APPLY "
    "TO LEAD' section of src/baselines/lead.py's module docstring for the "
    "full argument. Summary: (1) technical -- resolve_effective_min_words "
    "relaxes toward maximum_feasible_words, an arbitrary-subset bitset "
    "subset-sum, but Lead can only take a reading-order prefix, so the "
    "relaxed [min_words, max_words] window can be structurally unreachable "
    "for a prefix even when it is reachable for some other subset of the "
    "same sentences; (2) methodological -- ACTION_PLAN.md item 1e "
    "introduced min_words=200 to stop a mean-salience objective "
    "degenerating to a single sentence, and Lead has no such objective to "
    "degenerate. Empirical distribution: see "
    "docs/research/CODE_AUDIT_IEEE_Access.md finding F-16."
)


def _group_by_source_order(
    eligible_records: List[Dict[str, Any]]
) -> Dict[Any, List[int]]:
    """Map each source document to the (ascending) relative indices in it.

    ``eligible_records`` is already in ascending original-index order (see
    ``flatten_sentence_records``), so appending in iteration order preserves
    each document's internal sentence order. Legacy flat rows carry no
    ``source_order`` field at all; they are treated as one document (key
    ``0``), which makes ``round_robin`` degenerate to ``document_order`` for
    genuinely single-document input -- there is nothing to interleave.
    """

    groups: Dict[Any, List[int]] = {}
    for relative_index, record in enumerate(eligible_records):
        key = record.get("source_order", 0)
        groups.setdefault(key, []).append(relative_index)
    return groups


def _select_document_order(
    eligible_records: List[Dict[str, Any]], evaluator: SelectionObjective
) -> List[int]:
    selected: List[int] = []
    for relative_index in range(len(eligible_records)):
        if not evaluator.can_add(selected, relative_index):
            break
        selected.append(relative_index)
    return selected


def _select_round_robin(
    eligible_records: List[Dict[str, Any]], evaluator: SelectionObjective
) -> List[int]:
    groups = _group_by_source_order(eligible_records)
    queues = {key: list(indices) for key, indices in sorted(groups.items(), key=lambda kv: kv[0])}
    order = list(queues.keys())
    selected: List[int] = []
    made_progress = True
    while made_progress:
        made_progress = False
        for key in order:
            queue = queues[key]
            if not queue:
                continue
            candidate = queue[0]
            if not evaluator.can_add(selected, candidate):
                return selected
            queue.pop(0)
            selected.append(candidate)
            made_progress = True
    return selected


def _select_fabbri_first_k(
    eligible_records: List[Dict[str, Any]],
    evaluator: SelectionObjective,
    *,
    first_k: int,
) -> List[int]:
    """First ``first_k`` eligible sentences of every source document.

    ``can_add`` is unconditionally True today: ``_resolve_ordering`` only
    ever invokes this ordering with ``length_gate=False``, so the evaluator's
    ``max_length``/``max_sentences`` are both None and the sole remaining way
    ``can_add`` could return False -- a duplicate index -- never happens
    across per-document slices. The ``break`` below is therefore dead code
    right now; it exists only so the stopping rule reads the same as
    ``_select_document_order``/``_select_round_robin``, not because it
    currently changes any output.

    This ordering's semantics are per-document top-k, not a length-gated
    prefix: Fabbri et al. 2019's own definition has no total-length budget
    at all, it always takes exactly ``first_k`` sentences from every
    document regardless of how long the concatenation ends up. If
    ``length_gate`` is ever turned on for this ordering, ``break`` would
    start silently truncating a document's ``first_k`` quota whenever the
    *global* budget is tight -- at that point this is no longer Fabbri's
    definition, it is a length-gated prefix wearing its name. Enabling
    ``length_gate`` here requires first deciding whether the result should
    still be called ``fabbri_first_k``.
    """
    groups = _group_by_source_order(eligible_records)
    selected: List[int] = []
    for key in sorted(groups.keys(), key=lambda value: value):
        for relative_index in groups[key][:first_k]:
            if not evaluator.can_add(selected, relative_index):
                break
            selected.append(relative_index)
    return selected


def _resolve_ordering(ordering: str, first_k: int) -> Tuple[SelectFn, bool]:
    """Return (select_fn, length_gate) for a named ordering."""

    if ordering == "document_order":
        return _select_document_order, True
    if ordering == "round_robin":
        return _select_round_robin, True
    if ordering == "fabbri_first_k":
        def select_fn(eligible_records, evaluator):
            return _select_fabbri_first_k(eligible_records, evaluator, first_k=first_k)

        return select_fn, False
    raise ValueError(f"unknown lead ordering {ordering!r}; choose one of {ORDERINGS}")


def summarize_one_lead(
    doc: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    ordering: str = "document_order",
    first_k: int = 3,
) -> Dict[str, Any]:
    """Lead-baseline analogue of ``select_sentences.summarize_one``.

    ``min_words`` is never enforced for Lead -- see this module's
    ``LEAD_MIN_WORDS_NOT_APPLIED_REASON`` and the "MIN_WORDS DOES NOT APPLY
    TO LEAD" section of the module docstring.
    """

    select_fn, length_gate = _resolve_ordering(ordering, first_k)
    return summarize_one_baseline(
        doc,
        cfg,
        method=f"lead_{ordering}",
        select_fn=select_fn,
        length_gate=length_gate,
        apply_min_words=False,
        min_words_not_applied_reason=LEAD_MIN_WORDS_NOT_APPLIED_REASON,
    )
