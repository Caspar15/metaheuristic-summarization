from typing import List, Tuple, Optional
import numpy as np
import random

from src.selection.length_controller import will_fit_unit


def _count_tokens(s: str) -> int:
    return len(s.split())


def _objective(selected: List[int], base_scores: List[float], sim_mat: Optional[np.ndarray], alpha: float) -> float:
    if not selected:
        return -1e9
    importance = sum(base_scores[i] for i in selected)
    redundancy = 0.0
    if sim_mat is not None:
        for a in range(len(selected)):
            ia = selected[a]
            for b in range(a + 1, len(selected)):
                ib = selected[b]
                redundancy += float(sim_mat[ia, ib])
    return importance - (1 - alpha) * redundancy


def _construct_greedy_randomized(
    sentences: List[str],
    base_scores: List[float],
    sim_mat: Optional[np.ndarray],
    max_tokens: int,
    alpha: float,
    rcl_ratio: float,
    rng: random.Random,
    unit: str = "tokens",
    max_sentences: int | None = None,
) -> List[int]:
    selected: List[int] = []
    current_texts: List[str] = []
    remaining = set(range(len(sentences)))
    while remaining:
        cand = []
        util = []
        for i in list(remaining):
            if not will_fit_unit(current_texts, sentences[i], unit=unit, max_tokens=max_tokens, max_sentences=max_sentences):
                continue
            max_sim = float(np.max(sim_mat[i, selected])) if (selected and sim_mat is not None) else 0.0
            score = alpha * base_scores[i] - (1 - alpha) * max_sim
            cand.append(i)
            util.append(score)
        if not cand:
            break
        k = max(1, int(len(cand) * rcl_ratio))
        order = sorted(range(len(cand)), key=lambda t: util[t], reverse=True)
        rcl = [cand[idx] for idx in order[:k]]
        choice = rng.choice(rcl)
        selected.append(choice)
        current_texts.append(sentences[choice])
        remaining.remove(choice)
        # 是否還能再裝？若不行提前停止
        any_fit = any(
            (j in remaining)
            and will_fit_unit(current_texts, sentences[j], unit=unit, max_tokens=max_tokens, max_sentences=max_sentences)
            for j in remaining
        )
        if not any_fit:
            break
    selected.sort()
    return selected


def _local_search(
    solution: List[int],
    sentences: List[str],
    base_scores: List[float],
    sim_mat: Optional[np.ndarray],
    max_tokens: int,
    alpha: float,
    max_iter: int = 100,
    unit: str = "tokens",
    max_sentences: int | None = None,
) -> List[int]:
    if not solution:
        return solution
    selected = solution[:]
    best_val = _objective(selected, base_scores, sim_mat, alpha)
    n = len(sentences)
    it = 0
    while it < max_iter:
        it += 1
        improved = False
        # 1) 嘗試 swap: 選中 vs 未選中
        for i in selected[:]:
            for j in range(n):
                if j in selected:
                    continue
                # 檢查長度可行：先移除 i，再嘗試加入 j
                temp_sel = [k for k in selected if k != i]
                cur_texts = [sentences[k] for k in temp_sel]
                if not will_fit_unit(cur_texts, sentences[j], unit=unit, max_tokens=max_tokens, max_sentences=max_sentences):
                    continue
                cand = sorted(temp_sel + [j])
                val = _objective(cand, base_scores, sim_mat, alpha)
                if val > best_val + 1e-9:
                    selected = cand
                    best_val = val
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # 2) 嘗試 add：加入一個未選中
        cur_texts = [sentences[k] for k in selected]
        for j in range(n):
            if j in selected:
                continue
            if not will_fit_unit(cur_texts, sentences[j], unit=unit, max_tokens=max_tokens, max_sentences=max_sentences):
                continue
            cand = sorted(selected + [j])
            val = _objective(cand, base_scores, sim_mat, alpha)
            if val > best_val + 1e-9:
                selected = cand
                best_val = val
                improved = True
                break
        if improved:
            continue
        # 3) 嘗試 drop：移除一個已選中
        if len(selected) > 1:
            for i in selected[:]:
                cand = [k for k in selected if k != i]
                val = _objective(cand, base_scores, sim_mat, alpha)
                if val > best_val + 1e-9:
                    selected = sorted(cand)
                    best_val = val
                    improved = True
                    break
        if not improved:
            break
    return selected


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
) -> List[int]:
    """GRASP: 多輪建構 + 局部搜尋，回傳全程最佳解。

    - 建構：以 alpha 調和重要性與冗餘（MMR 型效用），用 RCL 隨機化加入。
    - 局部搜尋：套用 swap/add/drop 直到無改善或達迭代上限。
    """
    rng = random.Random(seed)
    best: List[int] = []
    best_val = -1e9
    for _ in range(max(1, iters)):
        sol = _construct_greedy_randomized(
            sentences, base_scores, sim_mat, max_tokens, alpha, rcl_ratio, rng, unit=unit, max_sentences=max_sentences
        )
        sol = _local_search(
            sol, sentences, base_scores, sim_mat, max_tokens, alpha, max_iter=200, unit=unit, max_sentences=max_sentences
        )
        val = _objective(sol, base_scores, sim_mat, alpha)
        if val > best_val:
            best = sol
            best_val = val
    return sorted(best)
