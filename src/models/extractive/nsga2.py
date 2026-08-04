from typing import Dict, List, Optional
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

from src.objectives.evaluator import (
    InfeasibleSelectionError,
    ObjectiveWeights,
    SelectionConstraints,
    SelectionEvaluation,
    SelectionObjective,
)


# --------------- problem definition ---------------

class SummarizationProblem(ElementwiseProblem):
    def __init__(
        self,
        evaluator: SelectionObjective,
    ):
        self.evaluator = evaluator
        n = len(evaluator.sentences)
        super().__init__(n_var=n, n_obj=3, n_constr=4, xl=0, xu=1, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        idx = np.where(x > 0)[0]
        evaluation = self.evaluator.evaluate(idx)
        out["F"] = [
            -evaluation.salience,
            -evaluation.facility_coverage,
            evaluation.redundancy,
        ]
        out["G"] = self.evaluator.inequality_constraints(idx)


# --------------- public API ---------------

def _least_violating_evaluation(
    evaluator: SelectionObjective,
    vectors,
) -> SelectionEvaluation:
    """Return the best population member actually attempted by NSGA-II."""

    # The empty subset is always upper-bound safe and is preferable to
    # emitting an attempted solution that itself exceeds max_length or
    # max_sentences.  Population members compete only if those hard upper
    # bounds are also satisfied.
    best: SelectionEvaluation | None = evaluator.evaluate([])
    best_key: tuple | None = None
    if best is not None:
        best_key = (
            sum(max(0.0, value) for value in best.violations.values()),
            -best.scalar_utility,
            tuple(best.selected_indices),
        )
    if vectors is not None:
        for vector in np.atleast_2d(vectors):
            indices = np.where(vector > 0)[0]
            evaluation = evaluator.evaluate(indices)
            if (
                evaluation.violations["max_length"] > 0
                or evaluation.violations["max_sentences"] > 0
            ):
                continue
            key = (
                sum(max(0.0, value) for value in evaluation.violations.values()),
                -evaluation.scalar_utility,
                tuple(evaluation.selected_indices),
            )
            if best_key is None or key < best_key:
                best = evaluation
                best_key = key
    assert best is not None
    return best

def nsga2_select(
    sentences: List[str],
    importance: List[float],
    sim_mat: np.ndarray,
    max_tokens: int,
    lambda_importance: float = 1.0,
    lambda_coverage: float = 0.8,
    lambda_redundancy: float = 0.7,
    unit: str = "tokens",
    max_sentences: int | None = None,
    pop_size: int = 100,
    n_gen: int = 100,
    seed: Optional[int] = None,
    coverage_method: str = "max",
    importance_aggregation: str = "sum",
    min_words: int = 0,
    require_nonempty: bool = True,
    evaluator: SelectionObjective | None = None,
    diagnostics: Optional[Dict] = None,
) -> List[int]:
    n = len(sentences)
    if n == 0:
        return []
    if pop_size < 2 or n_gen < 1:
        raise ValueError("NSGA-II requires pop_size >= 2 and n_gen >= 1")

    if evaluator is None:
        evaluator = SelectionObjective(
            sentences,
            importance,
            sim_mat,
            importance_aggregation=importance_aggregation,
            coverage_method=coverage_method,
            weights=ObjectiveWeights(
                salience=lambda_importance,
                facility_coverage=lambda_coverage,
                redundancy=lambda_redundancy,
            ),
            constraints=SelectionConstraints(
                length_unit=unit,
                max_length=max_tokens,
                min_words=min_words,
                max_sentences=max_sentences,
                require_nonempty=require_nonempty,
            ),
        )
    problem = SummarizationProblem(evaluator)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )
    if res.X is None:
        population_x = None if res.pop is None else res.pop.get("X")
        raise InfeasibleSelectionError(
            "NSGA-II returned no feasible Pareto solution",
            _least_violating_evaluation(evaluator, population_x),
            reason_code="optimizer_no_feasible_solution",
        )

    X = np.atleast_2d(res.X)
    best_val = -1e18
    best_idx = -1
    best_front_idx = -1
    pareto_front = []

    for i, x in enumerate(X):
        idx = np.where(x > 0)[0]
        evaluation = evaluator.evaluate(idx)
        if not evaluation.feasible:
            continue
        val = evaluation.scalar_utility
        pareto_front.append(evaluation.to_dict())
        if val > best_val + 1e-12 or (
            abs(val - best_val) <= 1e-12
            and (best_idx < 0 or tuple(idx.tolist()) < tuple(np.where(X[best_idx] > 0)[0].tolist()))
        ):
            best_val = val
            best_idx = i
            best_front_idx = len(pareto_front) - 1

    if best_idx >= 0:
        chosen = X[best_idx]
        sel = np.where(chosen > 0)[0].tolist()
        sel.sort()
        evaluator.assert_feasible(sel)
        if diagnostics is not None:
            diagnostics.update(
                {
                    "method": "nsga2",
                    "pareto_policy": "weighted_sum_on_shared_objectives",
                    "pareto_front": pareto_front,
                    "pareto_size": len(pareto_front),
                    "selected_pareto_row": best_front_idx,
                    "search": {
                        "population_size": int(pop_size),
                        "generations": int(n_gen),
                        "seed": seed,
                        "sampling": "BinaryRandomSampling",
                        "crossover": "TwoPointCrossover",
                        "mutation": "BitflipMutation",
                        "eliminate_duplicates": True,
                    },
                    "selection_weights": {
                        "salience": evaluator.weights.salience,
                        "facility_coverage": evaluator.weights.facility_coverage,
                        "redundancy": evaluator.weights.redundancy,
                    },
                }
            )
        return sel
    else:
        raise InfeasibleSelectionError(
            "NSGA-II returned no feasible summary",
            _least_violating_evaluation(evaluator, X),
            reason_code="optimizer_no_feasible_solution",
        )


if __name__ == "__main__":
    sentences = [
        "The cat sits on the mat.",
        "Dogs are loyal animals.",
        "Artificial intelligence is transforming the world.",
        "The quick brown fox jumps over the lazy dog.",
        "Data science is an interdisciplinary field.",
    ]
    importance = [0.8, 0.6, 0.9, 0.5, 0.7]

    rng = np.random.default_rng(42)
    sim_mat = rng.random((len(sentences), len(sentences)))
    sim_mat = (sim_mat + sim_mat.T) / 2
    np.fill_diagonal(sim_mat, 1.0)

    max_tokens = 12
    selected = nsga2_select(sentences, importance, sim_mat, max_tokens)

    print("Selected indices:", selected)
    print("Selected sentences:")
    for i in selected:
        print(f"- {sentences[i]}")
