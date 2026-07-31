"""Shared run contract for extractive baselines (Lead, and future baselines).

A baseline is only a valid reality check against a system run if it is
length-matched through the *same* feasibility contract -- see
``src.objectives.evaluator`` (``resolve_effective_min_words`` /
``resolve_selection_eligibility``) and CLAUDE.md section 2, finding 1: the
legacy Multi-News Lead-vs-system comparison was invalid in part because the
comparison was never actually run on an apples-to-apples length budget. This
module exists so every baseline goes through that one contract instead of
re-deriving its own approximation of it.

Every baseline built on top of ``summarize_one_baseline`` produces a
``predictions.jsonl`` row that is structurally interchangeable with
``src.pipeline.select_sentences.summarize_one`` output: same top-level keys,
same ``output_budget``/``selection_evaluation`` shape, so it can be scored by
``src.pipeline.evaluate`` and tabulated next to system runs without special
casing. Fields that only make sense for the candidate-pool/optimizer
machinery (``candidate_records``, ``optimizer_diagnostics``, route
provenance) are present but explicitly empty/None rather than omitted, so
downstream code reading those keys does not KeyError.

``objective_spec.status`` for every baseline is ``"baseline"``, never
``"task_profiled_v1"`` or ``"legacy_unprofiled"`` -- those two statuses are
reserved for methods that actually optimise the shared salience / facility
coverage / redundancy utility (see ``src.objectives.factory``); a baseline
does not, and labelling it otherwise would misrepresent what the artifact
records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.data.schemas import flatten_sentence_records
from src.objectives.evaluator import (
    ObjectiveWeights,
    SelectionConstraints,
    SelectionObjective,
    resolve_effective_min_words,
    resolve_selection_eligibility,
)
from src.pipeline.select_sentences import validate_experiment_document

# SelectFn receives the sentence records eligible under the active length
# budget (already in original document order, i.e. ascending original_index)
# and a SelectionObjective built over exactly those eligible sentences with
# every utility weight at zero -- baselines do not optimise salience/coverage
# /redundancy, they only need the same hard-bound feasibility gate every
# optimizer goes through. It must return indices *relative to* the eligible
# list, and must use ``evaluator.can_add`` to respect upper bounds rather than
# re-deriving them.
SelectFn = Callable[[List[Dict[str, Any]], SelectionObjective], List[int]]


@dataclass(frozen=True)
class LengthBudget:
    """The subset of ``length_control`` a baseline needs to be length-matched.

    Field names and defaults mirror ``src.pipeline.select_sentences.
    summarize_one`` (:lines 172-188) exactly, on purpose: a baseline and a
    system run sharing the same config's ``length_control`` block must reach
    identical values here.
    """

    unit: str
    selector_budget: int
    max_sentences: Optional[int]
    requested_min_words: int
    require_nonempty: bool


def resolve_length_budget(cfg: Mapping[str, Any]) -> LengthBudget:
    """Read ``length_control`` the same way a governed system run does."""

    lc = cfg.get("length_control", {}) or {}
    unit = (lc.get("unit", "tokens") or "tokens").lower()
    if unit not in {"words", "tokens", "sentences"}:
        raise ValueError("length_control.unit must be words, tokens, or sentences")
    max_tokens = int(lc.get("max_tokens", 100))
    max_words = int(lc.get("max_words", 400))
    selector_budget = max_words if unit == "words" else max_tokens
    max_sents_limit = lc.get("max_sentences", None)
    max_sents = int(max_sents_limit) if max_sents_limit is not None else None
    requested_min_words = int(lc.get("min_words", 0))
    require_nonempty = bool(lc.get("require_nonempty", True))
    if requested_min_words < 0:
        raise ValueError("length_control.min_words cannot be negative")
    if unit in {"words", "tokens"} and requested_min_words > selector_budget:
        raise ValueError(
            "length_control.min_words cannot exceed the active maximum length"
        )
    return LengthBudget(
        unit=unit,
        selector_budget=selector_budget,
        max_sentences=max_sents,
        requested_min_words=requested_min_words,
        require_nonempty=require_nonempty,
    )


def _baseline_objective_spec(method: str, task_profile: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    profile = task_profile or {}
    return {
        "status": "baseline",
        "method": method,
        "input_mode": profile.get("input_mode"),
        "output_mode": profile.get("output_mode"),
        "applicable": [],
        "active": [],
        "importance_aggregation": None,
        "coverage_method": None,
        "coverage_scope": "not_applicable",
        "weights": {"salience": 0.0, "facility_coverage": 0.0, "redundancy": 0.0},
        "required_max_sentences": None,
        "group_coverage": False,
    }


def _empty_candidate_pool(ineligible_sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "enabled": False,
        "configured_sources": [],
        "route_top_k": None,
        "min_per_route": None,
        "total_cap": None,
        "actual_size": 0,
        "coverage_guard": {},
        "selector_salience_source": None,
        "route_proposals": {},
        "allocation": {"actual_size": 0},
        "selection_ineligible_sentences": ineligible_sentences,
    }


def summarize_one_baseline(
    doc: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    method: str,
    select_fn: SelectFn,
    length_gate: bool = True,
    apply_min_words: bool = True,
    min_words_not_applied_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Baseline analogue of ``select_sentences.summarize_one``.

    ``length_gate`` is the one explicit flag for whether ``length_control``'s
    upper bounds are applied at all: ``max_words``/``max_tokens`` and
    ``max_sentences`` are either all enforced (``length_gate=True``, through
    the exact same ``resolve_selection_eligibility`` a system run uses -- not
    a reimplementation of it) or none of them are (``length_gate=False``).
    There is no separate per-field flag for the upper bounds because nothing
    in this module currently applies them independently; if a future
    ordering needs to enforce some of them but not others, ``length_gate``
    must be split rather than re-overloaded.

    Regardless of ``length_gate``, ``output_budget`` always records
    ``requested_max_words``/``requested_max_tokens``/``requested_max_sentences``
    -- what ``length_control`` actually asked for -- separately from
    ``max_words``/``max_tokens``/``max_sentences``, which are the *applied*
    values (``None`` when ``length_gate`` is False). A config's request must
    never disappear from the artifact just because a given ordering does not
    apply it.

    The lower bound (``min_words``) has its own, narrower toggle,
    ``apply_min_words``, because Lead needs to disable *only* the floor while
    still enforcing the ceiling (``length_gate=True``, ``apply_min_words=
    False``): when True (the default), it is enforced through
    ``resolve_effective_min_words`` exactly like a system run; when False,
    ``SelectionConstraints.min_words`` is forced to 0 regardless of what
    ``length_control.min_words`` requests. ``resolve_effective_min_words`` is
    called whenever ``length_gate`` is True, independent of
    ``apply_min_words`` -- i.e. "both cases" means both values of
    ``apply_min_words``, not both values of ``length_gate`` -- so
    ``requested_min_words`` and ``source_capacity_words`` are always
    populated whenever the upper bounds are active, even if the floor itself
    is not. ``min_words_not_applied_reason`` is required (fail loud, not
    silently omitted) whenever ``apply_min_words`` is False and
    ``length_gate`` is True, and is recorded verbatim as
    ``output_budget.min_words_not_applied_reason``.

    When ``length_gate`` is False, no word/sentence budget is enforced at
    all (only ``require_nonempty``); this only exists for baseline variants
    that are explicitly *not* comparable to a fixed-budget system run (see
    ``lead.py``'s ``fabbri_first_k`` mode) and every such row must be
    unmistakably labelled as a diagnostic, never plotted next to a
    length-matched Gate 2 comparison. ``resolve_effective_min_words`` is not
    called in this branch at all (there is no active upper bound to compute
    a subset-sum capacity against), so ``source_capacity_words`` and
    ``candidate_capacity_words`` are ``None`` here -- not a dropped request,
    but a genuinely undefined quantity: "how many words fit under a cap"
    has no answer when no cap applies. ``requested_min_words`` is still
    populated directly from ``length_control.min_words`` (that part never
    needed ``resolve_effective_min_words`` to begin with).
    """

    if length_gate and not apply_min_words and min_words_not_applied_reason is None:
        raise ValueError(
            "apply_min_words=False requires an explicit min_words_not_applied_reason "
            "so the artifact never silently drops why the requested floor was skipped"
        )

    validate_experiment_document(cfg, doc)
    sentence_records = flatten_sentence_records(doc)
    sentences = [record["text"] for record in sentence_records]
    budget = resolve_length_budget(cfg)

    # What length_control actually asked for, independent of length_gate --
    # computed once so both branches below read the same values rather than
    # risking two independently hand-written copies drifting apart.
    requested_max_words = budget.selector_budget if budget.unit == "words" else None
    requested_max_tokens = budget.selector_budget if budget.unit == "tokens" else None
    requested_max_sentences = budget.max_sentences

    if length_gate:
        min_words_resolution = resolve_effective_min_words(
            sentences,
            requested_min_words=budget.requested_min_words,
            max_length=budget.selector_budget,
            length_unit=budget.unit,
            max_sentences=budget.max_sentences,
        )
        eligibility = resolve_selection_eligibility(
            sentences,
            sentence_records,
            max_length=budget.selector_budget,
            length_unit=budget.unit,
            require_nonempty=budget.require_nonempty,
            document_id=doc.get("id"),
        )
        eligible_indices = eligibility.eligible_indices
        ineligible_sentences = eligibility.ineligible_sentences
        constraint_max_length: Optional[int] = budget.selector_budget
        constraint_max_sentences = budget.max_sentences
        if apply_min_words:
            effective_min_words = min_words_resolution.effective_min_words
            min_words_relaxed = min_words_resolution.min_words_relaxed
            relaxation_reason = min_words_resolution.relaxation_reason
            min_words_applied = True
            resolved_not_applied_reason = None
        else:
            effective_min_words = 0
            min_words_relaxed = False
            relaxation_reason = None
            min_words_applied = False
            resolved_not_applied_reason = min_words_not_applied_reason
        output_budget_length_fields = {
            "max_words": requested_max_words,
            "max_tokens": requested_max_tokens,
            "max_sentences": requested_max_sentences,
            "requested_max_words": requested_max_words,
            "requested_max_tokens": requested_max_tokens,
            "requested_max_sentences": requested_max_sentences,
            "min_words": effective_min_words,
            "requested_min_words": min_words_resolution.requested_min_words,
            "effective_min_words": effective_min_words,
            "source_capacity_words": min_words_resolution.source_capacity_words,
            "candidate_capacity_words": min_words_resolution.source_capacity_words,
            "min_words_relaxed": min_words_relaxed,
            "relaxation_reason": relaxation_reason,
            "min_words_applied": min_words_applied,
            "min_words_not_applied_reason": resolved_not_applied_reason,
        }
    else:
        eligible_indices = list(range(len(sentences)))
        ineligible_sentences = []
        constraint_max_length = None
        constraint_max_sentences = None
        effective_min_words = 0
        output_budget_length_fields = {
            # Applied values: None because length_gate is False, no upper
            # bound of any kind is enforced for this ordering (see
            # lead.py's fabbri_first_k docstring). The single explicit
            # "is this applied" marker is output_budget.length_gate itself,
            # not the presence/absence of these fields.
            "max_words": None,
            "max_tokens": None,
            "max_sentences": None,
            # Requested values: always populated from length_control,
            # regardless of whether this ordering applies them -- a config
            # asking for max_words=250 must still show up here even though
            # fabbri_first_k never enforces it.
            "requested_max_words": requested_max_words,
            "requested_max_tokens": requested_max_tokens,
            "requested_max_sentences": requested_max_sentences,
            "min_words": 0,
            "requested_min_words": budget.requested_min_words,
            "effective_min_words": 0,
            # Undefined, not dropped: "words that fit under a cap" has no
            # answer when no cap applies (resolve_effective_min_words /
            # maximum_feasible_words are never called in this branch).
            "source_capacity_words": None,
            "candidate_capacity_words": None,
            "min_words_relaxed": False,
            "relaxation_reason": None,
            "min_words_applied": False,
            "min_words_not_applied_reason": (
                "length_gate is disabled entirely for this diagnostic ordering "
                "(no word/sentence budget of any kind is enforced, min_words "
                "included); see src/baselines/lead.py's fabbri_first_k docstring"
            ),
        }

    eligible_records = [sentence_records[index] for index in eligible_indices]
    eligible_sentences = [sentences[index] for index in eligible_indices]

    evaluator = SelectionObjective(
        eligible_sentences,
        np.zeros(len(eligible_sentences)),
        None,
        importance_aggregation="mean",
        coverage_method="max",
        weights=ObjectiveWeights(salience=0.0, facility_coverage=0.0, redundancy=0.0),
        constraints=SelectionConstraints(
            length_unit=budget.unit,
            max_length=constraint_max_length,
            min_words=effective_min_words,
            max_sentences=constraint_max_sentences,
            require_nonempty=budget.require_nonempty,
        ),
    )

    picked_relative = select_fn(eligible_records, evaluator) if eligible_records else []
    selected = sorted(eligible_indices[index] for index in picked_relative)

    selection_evaluation = None
    if eligible_sentences:
        evaluation = evaluator.assert_feasible(picked_relative)
        selection_evaluation = evaluation.to_dict()
        selection_evaluation["candidate_relative_indices"] = list(
            selection_evaluation["selected_indices"]
        )
        selection_evaluation["selected_indices"] = list(selected)

    summary_sentences = [sentences[index] for index in selected]
    summary = "\n".join(summary_sentences)
    selected_sentences = [
        {**sentence_records[index], "selection_order": order, "selection_evidence": None}
        for order, index in enumerate(selected)
    ]

    return {
        "id": doc.get("id"),
        "selected_indices": selected,
        "selected_sentences": selected_sentences,
        "summary_sentences": summary_sentences,
        "summary": summary,
        "candidate_records": [],
        "candidate_pool": _empty_candidate_pool(ineligible_sentences),
        "objective_spec": _baseline_objective_spec(method, doc.get("task_profile")),
        "selection_evaluation": selection_evaluation,
        "optimizer_diagnostics": None,
        "output_budget": {
            "unit": budget.unit,
            "require_nonempty": budget.require_nonempty,
            "length_gate": length_gate,
            "selected_words": (
                selection_evaluation["selected_words"]
                if selection_evaluation is not None
                else 0
            ),
            **output_budget_length_fields,
        },
        "task_profile": doc.get("task_profile"),
    }
