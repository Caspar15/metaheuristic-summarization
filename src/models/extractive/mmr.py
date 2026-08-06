"""Deterministic Maximal Marginal Relevance sentence selection."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.objectives.evaluator import SelectionObjective


def mmr_select(
    sentences: List[str],
    relevance: List[float],
    similarity_matrix: np.ndarray,
    max_tokens: int,
    *,
    lambda_relevance: float = 0.7,
    unit: str = "tokens",
    max_sentences: Optional[int] = None,
    evaluator: SelectionObjective,
    diagnostics: Optional[Dict] = None,
    assert_feasible: bool = True,
) -> List[int]:
    """Select one sentence at a time using the canonical MMR rule.

    The method visits all candidates that still fit the active upper bounds;
    it does not stop merely because an MMR score becomes negative.  This is a
    fixed-budget summarization rule, not a relevance-threshold retrieval rule.
    ``evaluator`` owns every feasibility decision and the final shared-objective
    evaluation, so MMR cannot silently implement a different word counter.
    """

    del max_tokens, unit, max_sentences  # constraints are owned by evaluator
    if not sentences:
        return []
    if not 0.0 <= float(lambda_relevance) <= 1.0:
        raise ValueError("MMR lambda_relevance must be in [0, 1]")

    scores = np.asarray(relevance, dtype=float)
    similarities = np.asarray(similarity_matrix, dtype=float)
    n = len(sentences)
    if scores.shape != (n,):
        raise ValueError(f"MMR relevance must have shape ({n},)")
    if similarities.shape != (n, n):
        raise ValueError(f"MMR similarity matrix must have shape ({n}, {n})")
    if not np.all(np.isfinite(scores)):
        raise ValueError("MMR relevance contains non-finite values")
    if not np.all(np.isfinite(similarities)):
        raise ValueError("MMR similarity matrix contains non-finite values")

    selected_order: List[int] = []
    remaining = set(range(n))
    step_scores: List[Dict] = []
    weight = float(lambda_relevance)
    while remaining:
        ranked: List[tuple[float, int, float]] = []
        for candidate in sorted(remaining):
            if not evaluator.can_add(selected_order, candidate):
                continue
            redundancy = (
                0.0
                if not selected_order
                else float(np.max(similarities[candidate, selected_order]))
            )
            mmr_value = weight * scores[candidate] - (1.0 - weight) * redundancy
            ranked.append((float(mmr_value), candidate, redundancy))
        if not ranked:
            break
        # Stable scientific tie-break: lower candidate-relative index wins.
        mmr_value, candidate, redundancy = max(
            ranked, key=lambda item: (item[0], -item[1])
        )
        selected_order.append(candidate)
        remaining.remove(candidate)
        step_scores.append(
            {
                "step": len(selected_order),
                "candidate_relative_index": candidate,
                "relevance": float(scores[candidate]),
                "max_similarity_to_selected": redundancy,
                "mmr_score": mmr_value,
            }
        )

    if assert_feasible:
        evaluator.assert_feasible(selected_order)
    if diagnostics is not None:
        diagnostics.update(
            {
                "method": "mmr",
                "formula": (
                    "lambda*relevance-(1-lambda)*"
                    "max_similarity_to_selected"
                ),
                "lambda_relevance": weight,
                "tie_break": "lower_candidate_relative_index",
                "selection_order": list(selected_order),
                "steps": step_scores,
            }
        )
    return sorted(selected_order)
