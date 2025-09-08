# src/models/extractive/nsga2.py

from typing import List
import numpy as np

# 嘗試匯入 pymoo，如果不可用則提供 fallback
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.optimize import minimize
    from pymoo.core.problem import ElementwiseProblem
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    # 如果 pymoo 不可用，建立 dummy classes 避免程式碰到定義錯誤
    class ElementwiseProblem:
        def __init__(self, **kwargs):
            pass

class SummarizationProblem(ElementwiseProblem):
    """
    整合了 LengthController 的摘要問題定義
    """
    def __init__(self, sentences: List[str], importance: List[float], sim_mat: np.ndarray, 
                 length_controller_config: dict):
        self.sentences = sentences
        self.importance = np.array(importance)
        self.sim_mat = sim_mat
        self.length_config = length_controller_config
        
        # 根據 length_controller 設定決定約束類型
        self.unit = length_controller_config.get('unit', 'tokens')
        self.max_tokens = length_controller_config.get('max_tokens', 100)
        self.max_sentences = length_controller_config.get('max_sentences', 3)
        
        if not PYMOO_AVAILABLE:
            return
            
        n = len(sentences)
        super().__init__(n_var=n, n_obj=3, n_constr=1, xl=0, xu=1, type_var=int)

    def _count_tokens(self, text: str) -> int:
        return len(text.split())

    def _evaluate(self, x, out, *args, **kwargs):
        idx = np.where(x > 0)[0]
        
        # 目標 1: 重要性最大化 (取負號因為 pymoo 是最小化)
        imp = np.sum(self.importance[idx]) if idx.size > 0 else 0.0

        # 目標 2: 覆蓋度最大化 (與所有句子的最大相似度)
        if idx.size > 0 and self.sim_mat is not None:
            # 計算每個未選句子與已選句子的最大相似度
            coverage_scores = []
            for i in range(len(self.sentences)):
                if i not in idx:  # 未選句子
                    max_sim = np.max(self.sim_mat[i, idx]) if len(idx) > 0 else 0
                    coverage_scores.append(max_sim)
            cov = np.mean(coverage_scores) if coverage_scores else 0.0
        else:
            cov = 0.0

        # 目標 3: 冗餘度最小化 (已選句子間的平均相似度)
        if idx.size > 1 and self.sim_mat is not None:
            S = self.sim_mat[np.ix_(idx, idx)]
            red = np.mean(S[np.triu_indices(len(idx), k=1)])
        else:
            red = 0.0

        # 設定目標函數 (負號表示最大化)
        out["F"] = [-imp, -cov, red]
        
        # 約束：根據 unit 設定不同的約束
        if self.unit == 'sentences':
            # 句子數量約束：選中的句子數必須等於 max_sentences
            constraint = (len(idx) - self.max_sentences) ** 2
        else:  # tokens
            # Token 數量約束：選中句子的總 token 數不能超過 max_tokens
            total_tokens = sum(self._count_tokens(self.sentences[i]) for i in idx)
            constraint = max(0, total_tokens - self.max_tokens)
        
        out["G"] = [constraint]


class Nsga2Selector:
    """
    NSGA-II 多目標選句演算法，整合 LengthController
    """
    def __init__(self, config):
        self.lambda_importance = config.get("objectives", {}).get("lambda_importance", 1.0)
        self.lambda_coverage = config.get("objectives", {}).get("lambda_coverage", 0.8)
        self.lambda_redundancy = config.get("objectives", {}).get("lambda_redundancy", 0.7)
        
        # NSGA-II 特定參數
        nsga2_config = config.get("nsga2", {})
        self.pop_size = nsga2_config.get("pop_size", 50)
        self.n_gen = nsga2_config.get("n_gen", 100)
        self.seed = config.get("seed", 42)

    def select(self, sentences, scores, sim_matrix, length_controller):
        """
        執行 NSGA-II 選句，由傳入的 length_controller 控制長度。
        """
        # 檢查前置條件
        if not PYMOO_AVAILABLE:
            print("Warning: pymoo not available for NSGA-II. Falling back to greedy.")
            return self._fallback_greedy(sentences, scores, sim_matrix, length_controller)
        
        if sim_matrix is None:
            print("Warning: No similarity matrix available for NSGA-II. Falling back to greedy.")
            return self._fallback_greedy(sentences, scores, sim_matrix, length_controller)
        
        if len(sentences) == 0:
            return []

        # 從 length_controller 取得配置
        length_config = {
            'unit': length_controller.unit,
            'max_tokens': length_controller.max_tokens,
            'max_sentences': length_controller.max_sentences
        }

        # 定義問題
        problem = SummarizationProblem(sentences, scores, sim_matrix, length_config)

        # 定義演算法
        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(prob=0.9),
            mutation=BitflipMutation(prob=1.0 / len(sentences)),
            eliminate_duplicates=True,
        )

        # 執行優化
        try:
            res = minimize(problem, algorithm, ("n_gen", self.n_gen), verbose=False, seed=self.seed)
            
            if res.X is None or len(res.X) == 0:
                print("Warning: NSGA-II failed to find solutions. Falling back to greedy.")
                return self._fallback_greedy(sentences, scores, sim_matrix, length_controller)

            # 從 Pareto 前緣中選擇最佳解
            X = np.atleast_2d(res.X)
            F = np.atleast_2d(res.F)
            
            # 標準化目標函數值並使用線性加權
            F_norm = self._normalize_objectives(F)
            weights = np.array([self.lambda_importance, self.lambda_coverage, self.lambda_redundancy])
            scalar_scores = np.dot(F_norm, weights)
            
            # 選擇標量化分數最小的解 (因為目標都被轉為最小化)
            best_idx = np.argmin(scalar_scores)
            best_solution = X[best_idx]
            selected_indices = np.where(best_solution > 0)[0].tolist()
            
            return sorted(selected_indices)

        except Exception as e:
            print(f"Warning: NSGA-II optimization failed: {e}. Falling back to greedy.")
            return self._fallback_greedy(sentences, scores, sim_matrix, length_controller)

    def _normalize_objectives(self, F):
        """標準化目標函數值到 [0, 1] 範圍"""
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        # 避免除以零
        F_range = np.where(F_range == 0, 1, F_range)
        return (F - F_min) / F_range

    def _fallback_greedy(self, sentences, scores, sim_matrix, length_controller):
        """當 NSGA-II 無法使用時的 fallback 策略"""
        try:
            from src.models.extractive.greedy import GreedySelector
            greedy_config = {"redundancy": {"lambda": 0.7}}
            greedy = GreedySelector(greedy_config)
            return greedy.select(sentences, scores, sim_matrix, length_controller)
        except ImportError:
            print("Warning: GreedySelector also not available. Returning empty selection.")
            return []


# 原有的函數保持向後相容性 (可選)
def nsga2_select(
    sentences: List[str],
    importance: List[float],
    sim_mat: np.ndarray,
    max_tokens: int,
    lambda_importance: float = 1.0,
    lambda_coverage: float = 0.8,
    lambda_redundancy: float = 0.7,
) -> List[int]:
    """
    向後相容的函數介面（已棄用，建議使用 Nsga2Selector）
    """
    # 為了向後相容，我們創建一個臨時的 LengthController 和 Nsga2Selector
    try:
        from src.selection.length_controller import LengthController
        
        config = {
            "objectives": {
                "lambda_importance": lambda_importance,
                "lambda_coverage": lambda_coverage,
                "lambda_redundancy": lambda_redundancy
            }
        }
        length_controller = LengthController({"unit": "tokens", "max_tokens": max_tokens}, sentences)
        selector = Nsga2Selector(config)
        return selector.select(sentences, importance, sim_mat, length_controller)
    except Exception:
        return []


# ==============================================================================
# 測試區塊
# ==============================================================================
if __name__ == "__main__":
    # 路徑處理，讓腳本可以獨立運行
    import sys
    import os
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    sys.path.insert(0, project_root)
    
    from src.selection.length_controller import LengthController
    
    print("--- Testing Nsga2Selector ---")
    
    sentences = [
        "The cat sits on the mat.",
        "Dogs are loyal animals.", 
        "Artificial intelligence is transforming the world.",
        "The quick brown fox jumps over the lazy dog.",
        "Data science is an interdisciplinary field.",
    ]
    importance = [0.8, 0.6, 0.9, 0.5, 0.7]

    # 創建相似度矩陣
    rng = np.random.default_rng(42)
    sim_mat = rng.random((len(sentences), len(sentences)))
    sim_mat = (sim_mat + sim_mat.T) / 2
    np.fill_diagonal(sim_mat, 1.0)

    # 測試配置
    config = {
        "objectives": {
            "lambda_importance": 1.0,
            "lambda_coverage": 0.8,
            "lambda_redundancy": 0.7
        },
        "nsga2": {
            "pop_size": 20,
            "n_gen": 30
        },
        "seed": 42
    }

    # 創建 LengthController 和 Nsga2Selector
    length_controller = LengthController({"unit": "sentences", "max_sentences": 3}, sentences)
    selector = Nsga2Selector(config)

    # 執行選句
    selected = selector.select(sentences, importance, sim_mat, length_controller)

    print(f"Selected indices: {selected}")
    print("Selected sentences:")
    for i in selected:
        print(f"  - {sentences[i]}")

    # 驗證結果
    if PYMOO_AVAILABLE:
        assert len(selected) == 3, f"Expected 3 sentences, got {len(selected)}"
        print("Test PASSED: Selected exactly 3 sentences!")
    else:
        print("Test SKIPPED: pymoo not available, fallback was used.")
    
    print("\n--- Testing backward compatibility ---")
    # 測試舊的函數介面
    selected_old = nsga2_select(sentences, importance, sim_mat, max_tokens=15)
    print(f"Old interface selected: {selected_old}")
