from typing import List, Optional
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

from src.utils.tokenizer import count_tokens


# --------------- coverage helpers ---------------

def _coverage_max(sim_mat: np.ndarray, idx: np.ndarray) -> float:
    """Original: mean of max similarity to any selected sentence."""
    sub = sim_mat[:, idx]
    return float(np.mean(np.max(sub, axis=1)))


def _coverage_set(sim_mat: np.ndarray, idx: np.ndarray) -> float:
    """Set-coverage: marginal contribution (greedy submodular)."""
    n = sim_mat.shape[0]
    covered = np.zeros(n)
    for i in idx:
        covered = np.maximum(covered, sim_mat[:, i])
    return float(covered.mean())


def _coverage_diversity(sim_mat: np.ndarray, idx: np.ndarray) -> float:
    """Coverage minus internal redundancy penalty."""
    cov = _coverage_max(sim_mat, idx)
    if idx.size > 1:
        S = sim_mat[np.ix_(idx, idx)]
        red = float(np.mean(S[np.triu_indices(len(idx), k=1)]))
    else:
        red = 0.0
    return cov - 0.3 * red


_COVERAGE_FNS = {
    "max": _coverage_max,
    "set": _coverage_set,
    "diversity": _coverage_diversity,
}


def _compute_coverage(sim_mat: np.ndarray, idx: np.ndarray, method: str = "max") -> float:
    fn = _COVERAGE_FNS.get(method, _coverage_max)
    return fn(sim_mat, idx)


# --------------- problem definition ---------------

class SummarizationProblem(ElementwiseProblem):
    def __init__(
        self,
        sentences: List[str],
        importance: List[float],
        sim_mat: np.ndarray,
        max_tokens: int,
        unit: str = "tokens",
        max_sentences: Optional[int] = None,
        coverage_method: str = "max",
    ):
        self.sentences = sentences
        self.importance = np.array(importance)
        self.sim_mat = sim_mat
        self.max_tokens = max_tokens
        self.unit = (unit or "tokens").lower()
        self.max_sentences = max_sentences if (max_sentences is not None) else 10**9
        self.coverage_method = coverage_method
        n = len(sentences)
        super().__init__(n_var=n, n_obj=3, n_constr=1, xl=0, xu=1, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        idx = np.where(x > 0)[0]
        imp = np.sum(self.importance[idx]) if idx.size > 0 else 0.0

        cov = _compute_coverage(self.sim_mat, idx, self.coverage_method) if idx.size > 0 else 0.0

        if idx.size > 1:
            S = self.sim_mat[np.ix_(idx, idx)]
            red = np.mean(S[np.triu_indices(len(idx), k=1)])
        else:
            red = 0.0

        out["F"] = [-imp, -cov, red]
        if self.unit == "sentences":
            total_sents = len(idx)
            out["G"] = [total_sents - int(self.max_sentences)]
        else:
            total_tokens = sum(count_tokens(self.sentences[i]) for i in idx)
            out["G"] = [total_tokens - self.max_tokens]


# --------------- public API ---------------

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
) -> List[int]:
    n = len(sentences)
    if n == 0:
        return []

    problem = SummarizationProblem(
        sentences, importance, sim_mat, max_tokens,
        unit=unit, max_sentences=max_sentences,
        coverage_method=coverage_method,
    )

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
        return []

    X = np.atleast_2d(res.X)
    best_val = -1e18
    best_idx = -1

    for i, x in enumerate(X):
        idx = np.where(x > 0)[0]
        imp = np.sum(np.array(importance)[idx]) if idx.size > 0 else 0.0
        cov = _compute_coverage(sim_mat, idx, coverage_method) if idx.size > 0 else 0.0
        red = np.mean(sim_mat[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)]) if idx.size > 1 else 0.0

        val = lambda_importance * imp + lambda_coverage * cov - lambda_redundancy * red
        if val > best_val:
            best_val = val
            best_idx = i

    if best_idx >= 0:
        chosen = X[best_idx]
        sel = np.where(chosen > 0)[0].tolist()
        sel.sort()
        return sel
    else:
        return []


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
