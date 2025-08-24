from typing import List, Tuple
import numpy as np
import random


def _count_tokens(text: str) -> int:
    return len(text.split())


def _evaluate(
    mask: np.ndarray,
    sentences: List[str],
    importance: List[float],
    sim: np.ndarray,
) -> Tuple[float, float, float, int, int]:
    """回傳 (importance, coverage, redundancy, tokens, k)
    - importance: sum of importance scores (maximize)
    - coverage: mean of max-sim to selected for all sentences (maximize)
    - redundancy: mean pairwise sim among selected (minimize)
    - tokens: total tokens of selected
    - k: number of selected
    """
    idx = np.where(mask > 0.5)[0]
    k = int(idx.size)
    if k == 0:
        return 0.0, 0.0, 0.0, 0, 0
    imp = float(np.sum(np.array(importance)[idx]))
    # coverage: 對每個句子計算對已選集的最大相似度後取平均
    if k:
        sub = sim[:, idx]
        cov = float(np.mean(np.max(sub, axis=1)))
    else:
        cov = 0.0
    # redundancy: 已選集的平均兩兩相似度
    if k > 1:
        S = sim[np.ix_(idx, idx)]
        upper = np.triu_indices(k, k=1)
        pairs = S[upper]
        red = float(np.mean(pairs)) if pairs.size else 0.0
    else:
        red = 0.0
    toks = int(sum(_count_tokens(sentences[i]) for i in idx))
    return imp, cov, red, toks, k


def _repair(mask: np.ndarray, sentences: List[str], importance: List[float], max_tokens: int) -> np.ndarray:
    """若超出長度上限，依性價比（importance/tokens）移除句子直到滿足。"""
    idx = np.where(mask > 0.5)[0].tolist()
    if not idx:
        return mask
    def total_tokens(ids: List[int]) -> int:
        return sum(_count_tokens(sentences[i]) for i in ids)
    cur = idx[:]
    while total_tokens(cur) > max_tokens and cur:
        # remove min value per token
        ratios = [ (importance[i]/max(1,_count_tokens(sentences[i])), i) for i in cur ]
        ratios.sort()  # ascending -> remove worst
        cur.remove(ratios[0][1])
    out = np.zeros_like(mask)
    out[cur] = 1.0
    return out


def _fast_nondominated_sort(F: np.ndarray) -> List[List[int]]:
    # F: (pop_size, m_objectives) where all objectives are minimized
    pop = F.shape[0]
    S = [[] for _ in range(pop)]
    n = [0] * pop
    fronts: List[List[int]] = [[]]
    for p in range(pop):
        for q in range(pop):
            if p == q:
                continue
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def _crowding_distance(front_idx: List[int], F: np.ndarray) -> np.ndarray:
    if not front_idx:
        return np.array([])
    m = F.shape[1]
    Nf = len(front_idx)
    dist = np.zeros(Nf)
    for j in range(m):
        values = F[front_idx, j]
        order = np.argsort(values)
        dist[order[0]] = dist[order[-1]] = np.inf
        vmin, vmax = values[order[0]], values[order[-1]]
        denom = vmax - vmin if vmax > vmin else 1.0
        for k in range(1, Nf - 1):
            dist[order[k]] += (values[order[k + 1]] - values[order[k - 1]]) / denom
    return dist


def nsga2_select(
    sentences: List[str],
    importance: List[float],
    sim_mat: np.ndarray,
    max_tokens: int,
    lambda_importance: float = 1.0,
    lambda_coverage: float = 0.8,
    lambda_redundancy: float = 0.7,
) -> List[int]:
    """NSGA-II 標準流程（簡潔版）：
    - 目標：maximize importance, maximize coverage, minimize redundancy。
    - 可行性：透過修補（repair）確保不超過 max_tokens。
    - 回傳：以加權標量化（lambda_*）從最後族群 Pareto 前沿挑選單一代表解。
    """
    rng = random.Random(42)
    n = len(sentences)
    if n == 0:
        return []
    pop_size = max(20, min(2 * n, 60))
    n_gen = 30
    cx_prob = 0.9
    mut_prob = 1.0 / n if n > 0 else 0.1

    def random_ind() -> np.ndarray:
        mask = np.array([1.0 if rng.random() < 0.2 else 0.0 for _ in range(n)], dtype=float)
        mask = _repair(mask, sentences, importance, max_tokens)
        return mask

    def crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if rng.random() > cx_prob:
            return a.copy(), b.copy()
        child1 = a.copy()
        child2 = b.copy()
        for i in range(n):
            if rng.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def mutate(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        changed = False
        for i in range(n):
            if rng.random() < mut_prob:
                out[i] = 1.0 - out[i]
                changed = True
        if changed:
            out = _repair(out, sentences, importance, max_tokens)
        return out

    def evaluate_pop(pop: List[np.ndarray]):
        vals = []
        for m in pop:
            imp, cov, red, toks, k = _evaluate(m, sentences, importance, sim_mat)
            # 轉為最小化向量以供非支配排序 (-imp, -cov, red)
            vals.append((imp, cov, red, toks, k, np.array([-imp, -cov, red], dtype=float)))
        return vals

    # 初始化族群
    population = [random_ind() for _ in range(pop_size)]

    for _ in range(n_gen):
        # 評分與排序
        metrics = evaluate_pop(population)
        F = np.vstack([m[-1] for m in metrics])  # for sorting
        fronts = _fast_nondominated_sort(F)
        # 建立 mating pool by tournament selection (rank + crowding)
        ranks = np.zeros(len(population), dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = r
        # 擁擠距離（每個前沿單獨計算）
        distances = np.zeros(len(population))
        for front in fronts:
            if not front:
                continue
            d = _crowding_distance(front, F)
            for i, idx in enumerate(front):
                distances[idx] = d[i]

        def tournament(i: int, j: int) -> int:
            if ranks[i] < ranks[j]:
                return i
            if ranks[j] < ranks[i]:
                return j
            return i if distances[i] > distances[j] else j

        parents = []
        for _ in range(pop_size):
            i, j = rng.randrange(pop_size), rng.randrange(pop_size)
            parents.append(population[tournament(i, j)])

        # 交叉與突變
        offspring: List[np.ndarray] = []
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % pop_size]
            c1, c2 = crossover(p1, p2)
            offspring.append(mutate(c1))
            offspring.append(mutate(c2))

        # 精英保留：父子合併，依前沿與擁擠距離截斷
        population = population + offspring
        metrics = evaluate_pop(population)
        F = np.vstack([m[-1] for m in metrics])
        fronts = _fast_nondominated_sort(F)
        new_pop: List[np.ndarray] = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([population[i] for i in front])
            else:
                # 根據 crowding distance 選擇剩餘名額
                d = _crowding_distance(front, F)
                order = np.argsort(-d)  # 大到小
                for idx in order:
                    if len(new_pop) >= pop_size:
                        break
                    new_pop.append(population[front[idx]])
                break
        population = new_pop

    # 最後從族群挑代表解：使用加權標量化（lambda_*）
    best_idx = -1
    best_val = -1e18
    for idx, m in enumerate(population):
        imp, cov, red, toks, k = _evaluate(m, sentences, importance, sim_mat)
        val = lambda_importance * imp + lambda_coverage * cov - lambda_redundancy * red
        if val > best_val:
            best_val = val
            best_idx = idx
    chosen = population[best_idx]
    sel = np.where(chosen > 0.5)[0].tolist()
    sel.sort()
    return sel
