from typing import List, Optional
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

class SummarizationProblem(ElementwiseProblem):
    def __init__(
        self,
        sentences: List[str],
        importance: List[float],
        sim_mat: np.ndarray,
        max_tokens: int,
        unit: str = "tokens",
        max_sentences: Optional[int] = None,
    ):
        self.sentences = sentences
        self.importance = np.array(importance)
        self.sim_mat = sim_mat
        self.max_tokens = max_tokens
        self.unit = (unit or "tokens").lower()
        self.max_sentences = max_sentences if (max_sentences is not None) else 10**9
        n = len(sentences)
        super().__init__(n_var=n, n_obj=3, n_constr=1, xl=0, xu=1, type_var=int)

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def _evaluate(self, x, out, *args, **kwargs):
        idx = np.where(x > 0)[0]
        imp = np.sum(self.importance[idx]) if idx.size > 0 else 0.0

        if idx.size > 0:
            sub = self.sim_mat[:, idx]
            cov = np.mean(np.max(sub, axis=1))
        else:
            cov = 0.0

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
            total_tokens = sum(self._count_tokens(self.sentences[i]) for i in idx)
            out["G"] = [total_tokens - self.max_tokens]

def nsga2_select(
    sentences: List[str], # 句子列表
    importance: List[float], # 句子重要性分數
    sim_mat: np.ndarray, # 句子相似度矩陣   
    max_tokens: int, # 摘要的最大長度限制（ Token or sentence ）
    lambda_importance: float = 1.0, # 重要性權重
    lambda_coverage: float = 0.8, # 覆蓋率權重
    lambda_redundancy: float = 0.7, # 冗餘度權重
    unit: str = "tokens", # 長度單位 ("tokens" or "sentences")
    max_sentences: int | None = None, # 最大句子數限制 (僅在 unit="sentences" 時使用)
) -> List[int]:
    n = len(sentences)
    if n == 0:
        return []

    problem = SummarizationProblem(
        sentences, importance, sim_mat, max_tokens, unit=unit, max_sentences=max_sentences
    )

    algorithm = NSGA2(
        pop_size=max(20, min(2 * n, 60)),
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, ("n_gen", 30), verbose=False, seed=42)
    if res.X is None:
        return []

    X = np.atleast_2d(res.X)
    best_val = -1e18
    best_idx = -1

    for i, x in enumerate(X):
        idx = np.where(x > 0)[0]
        imp = np.sum(np.array(importance)[idx]) if idx.size > 0 else 0.0
        cov = np.mean(np.max(sim_mat[:, idx], axis=1)) if idx.size > 0 else 0.0
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

    print("選中的句子索引:", selected)
    print("選中的句子:")
    for i in selected:
        print(f"- {sentences[i]}")
