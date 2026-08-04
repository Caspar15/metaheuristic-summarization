"""GRASP search over the same objective contract as other selectors."""

from typing import List, Optional
import random

import numpy as np

from src.objectives.evaluator import (
    InfeasibleSelectionError,
    ObjectiveWeights,
    SelectionConstraints,
    SelectionEvaluation,
    SelectionObjective,
)


def _construct_greedy_randomized(
    evaluator: SelectionObjective,
    n: int,
    rcl_ratio: float,
    rng: random.Random,
) -> List[int]:
    selected: List[int] = []
    remaining = set(range(n))
    while remaining:
        current = evaluator.evaluate(selected)
        ranked: list[tuple[float, int]] = []
        for candidate in sorted(remaining):
            if evaluator.can_add(selected, candidate):
                value = evaluator.evaluate(selected + [candidate]).scalar_utility
                ranked.append((value, candidate))
        if not ranked:
            break
        ranked.sort(key=lambda item: (-item[0], item[1]))
        if selected and current.feasible and ranked[0][0] <= current.scalar_utility + 1e-12:
            break
        rcl_size = max(1, int(np.ceil(len(ranked) * rcl_ratio)))
        choice = rng.choice([candidate for _, candidate in ranked[:rcl_size]])
        selected.append(choice)
        remaining.remove(choice)
    return sorted(selected)


def _local_search(
    solution: List[int],
    evaluator: SelectionObjective,
    n: int,
    max_iter: int = 200,
    early_stop: int = 10,
) -> List[int]:
    selected = sorted(solution)
    if not evaluator.evaluate(selected).feasible:
        return selected
    best_value = evaluator.evaluate(selected).scalar_utility
    stale = 0
    for _ in range(max_iter):
        best_neighbor: List[int] | None = None
        best_neighbor_value = best_value
        selected_set = set(selected)

        # Deterministic neighborhood enumeration makes the seed affect only
        # randomized construction, not accidental set/dict ordering.
        for removed in selected:
            for added in range(n):
                if added in selected_set:
                    continue
                candidate = sorted([i for i in selected if i != removed] + [added])
                evaluation = evaluator.evaluate(candidate)
                if evaluation.feasible and evaluation.scalar_utility > best_neighbor_value + 1e-12:
                    best_neighbor = candidate
                    best_neighbor_value = evaluation.scalar_utility
        for added in range(n):
            if added in selected_set or not evaluator.can_add(selected, added):
                continue
            candidate = sorted(selected + [added])
            evaluation = evaluator.evaluate(candidate)
            if evaluation.feasible and evaluation.scalar_utility > best_neighbor_value + 1e-12:
                best_neighbor = candidate
                best_neighbor_value = evaluation.scalar_utility
        for removed in selected:
            candidate = [i for i in selected if i != removed]
            evaluation = evaluator.evaluate(candidate)
            if evaluation.feasible and evaluation.scalar_utility > best_neighbor_value + 1e-12:
                best_neighbor = candidate
                best_neighbor_value = evaluation.scalar_utility

        if best_neighbor is None:
            stale += 1
            if stale >= early_stop:
                break
        else:
            selected = best_neighbor
            best_value = best_neighbor_value
            stale = 0
    return sorted(selected)


def grasp_select(
    sentences: List[str],
    base_scores: List[float],
    sim_mat: Optional[np.ndarray],
    max_tokens: int,
    alpha: float = 0.7,
    iters: int = 20,
    rcl_ratio: float = 0.3,
    seed: int | None = None,
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
    if not sentences:
        return []
    if not 0 < rcl_ratio <= 1:
        raise ValueError("rcl_ratio must be in (0, 1]")
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

    rng = random.Random(seed)
    best: List[int] | None = None
    best_value = -np.inf
    best_infeasible: SelectionEvaluation | None = None
    best_infeasible_key: tuple | None = None
    stale_rounds = 0
    for _ in range(max(1, iters)):
        solution = _construct_greedy_randomized(
            evaluator, len(sentences), rcl_ratio, rng
        )
        construction_evaluation = evaluator.evaluate(solution)
        if not construction_evaluation.feasible:
            # Preserve the best solution GRASP actually attempted so the
            # caller can emit an auditable per-document infeasible row.  This
            # is not repair/backfill: feasible-run selection is unchanged.
            key = (
                sum(max(0.0, value) for value in construction_evaluation.violations.values()),
                -construction_evaluation.scalar_utility,
                tuple(construction_evaluation.selected_indices),
            )
            if best_infeasible_key is None or key < best_infeasible_key:
                best_infeasible = construction_evaluation
                best_infeasible_key = key
            continue
        solution = _local_search(solution, evaluator, len(sentences))
        value = evaluator.evaluate(solution).scalar_utility
        if value > best_value + 1e-12:
            best = solution
            best_value = value
            stale_rounds = 0
        else:
            stale_rounds += 1
            if stale_rounds >= max(3, iters // 3):
                break
    if best is None:
        attempted = best_infeasible or evaluator.evaluate([])
        raise InfeasibleSelectionError(
            "GRASP could not construct a feasible summary",
            attempted,
            reason_code="optimizer_no_feasible_solution",
        )
    evaluator.assert_feasible(best)
    return sorted(best)
