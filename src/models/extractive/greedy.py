"""Deterministic greedy search over the shared selection objective."""

from typing import List, Optional

import numpy as np

from src.objectives.evaluator import (
    ObjectiveWeights,
    SelectionConstraints,
    SelectionObjective,
)
from src.utils.tokenizer import count_tokens


def greedy_select(
    sentences: List[str],
    base_scores: List[float],
    sim_mat: Optional[np.ndarray],
    max_tokens: int,
    alpha: float = 0.7,
    unit: str = "tokens",
    max_sentences: int | None = None,
    *,
    evaluator: SelectionObjective | None = None,
    importance_aggregation: str = "sum",
    coverage_method: str = "max",
    lambda_importance: float | None = None,
    lambda_coverage: float = 0.0,
    lambda_redundancy: float | None = None,
    min_words: int = 0,
    require_nonempty: bool = True,
) -> List[int]:
    """Select a subset by deterministic marginal utility.

    ``alpha`` remains only as a backward-compatible way to derive salience and
    redundancy weights when explicit lambdas are not supplied.  Formal
    pipeline runs inject the same evaluator used by GRASP and NSGA-II.
    """

    if not sentences:
        return []
    if evaluator is None:
        evaluator = SelectionObjective(
            sentences,
            base_scores,
            sim_mat,
            importance_aggregation=importance_aggregation,
            coverage_method=coverage_method,
            weights=ObjectiveWeights(
                salience=(alpha if lambda_importance is None else lambda_importance),
                facility_coverage=lambda_coverage,
                redundancy=(
                    (1.0 - alpha)
                    if lambda_redundancy is None
                    else lambda_redundancy
                ),
            ),
            constraints=SelectionConstraints(
                length_unit=unit,
                max_length=max_tokens,
                min_words=min_words,
                max_sentences=max_sentences,
                require_nonempty=require_nonempty,
            ),
        )

    selected: List[int] = []
    remaining = set(range(len(sentences)))
    while remaining:
        current = evaluator.evaluate(selected)
        ranked: list[tuple[float, int]] = []
        constraints = evaluator.constraints
        eligible = sorted(remaining)
        # Preserve can_add()'s cheap structural guards before asking the
        # objective to form extensions.  This is required for single-item
        # objectives: a second item is structurally inadmissible and must be
        # skipped before single-item salience is evaluated.
        if (
            constraints.max_sentences is not None
            and len(selected) + 1 > constraints.max_sentences
        ):
            eligible = []
        if constraints.max_length is not None:
            if constraints.length_unit.lower() == "sentences":
                if len(selected) + 1 > constraints.max_length:
                    eligible = []
            else:
                eligible = [
                    candidate
                    for candidate in eligible
                    if current.selected_words + count_tokens(sentences[candidate])
                    <= constraints.max_length
                ]
        additions = evaluator.evaluate_additions(selected, eligible)
        for candidate, evaluation in additions.items():
            # Lower bounds may be violated during construction.  This is the
            # same upper-bound predicate as SelectionObjective.can_add(), but
            # uses the already-computed extension instead of evaluating it a
            # second time.
            if (
                evaluation.violations["max_length"] > 0
                or evaluation.violations["max_sentences"] > 0
            ):
                continue
            ranked.append((evaluation.scalar_utility, candidate))
        if not ranked:
            break
        # Stable scientific tie-break: lower candidate index wins.
        best_value, best_candidate = max(ranked, key=lambda item: (item[0], -item[1]))
        # Once all lower bounds are satisfied, do not add a sentence that
        # worsens the declared objective merely to fill the budget.
        if selected and current.feasible and best_value <= current.scalar_utility + 1e-12:
            break
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    evaluator.assert_feasible(selected)
    return sorted(selected)
