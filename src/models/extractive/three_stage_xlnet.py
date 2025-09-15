# three_stage_xlnet.py
import sys
import os

# 動態添加項目根目錄到 Python 路徑
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)

from typing import List, Dict
import numpy as np

# 導入現有的函數
from src.models.extractive.nsga2 import nsga2_select
from src.models.extractive.greedy import greedy_select
from src.models.extractive.xlnet_rank import XLNetRanker
from src.selection.length_controller import LengthController

# 創建 GreedySelector 適配器類
class GreedySelector:
    """適配器類：將現有的 greedy_select 函數包裝成與其他 Selector 一致的介面"""
    def __init__(self, config):
        self.config = config
        self.alpha = config.get("redundancy", {}).get("lambda", 0.7)
        
    def select(self, sentences, scores, sim_matrix, length_controller):
        try:
            selected = greedy_select(
                sentences=sentences,
                base_scores=scores,
                sim_mat=sim_matrix,
                max_tokens=length_controller.max_tokens,
                alpha=self.alpha,
                unit=length_controller.unit,
                max_sentences=length_controller.max_sentences
            )
            return selected
        except Exception as e:
            print(f"Greedy select failed: {e}, using simple fallback")
            return self._simple_fallback(sentences, scores, length_controller)
    
    def _simple_fallback(self, sentences, scores, length_controller):
        if not sentences:
            return []
        n = min(length_controller.max_sentences, len(sentences))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted(top_indices[:n])

# 創建 NSGA-II 適配器類
class Nsga2SelectorAdapter:
    """適配器類：將現有的 nsga2_select 函數包裝成與其他 Selector 一致的介面"""
    def __init__(self, config):
        self.config = config
        self.lambda_importance = config.get("objectives", {}).get("lambda_importance", 1.0)
        self.lambda_coverage = config.get("objectives", {}).get("lambda_coverage", 0.8)
        self.lambda_redundancy = config.get("objectives", {}).get("lambda_redundancy", 0.7)
        
    def select(self, sentences, scores, sim_matrix, length_controller):
        try:
            if length_controller.unit == "sentences":
                selected = nsga2_select(
                    sentences=sentences,
                    importance=scores,
                    sim_mat=sim_matrix,
                    max_tokens=100,
                    lambda_importance=self.lambda_importance,
                    lambda_coverage=self.lambda_coverage,
                    lambda_redundancy=self.lambda_redundancy,
                    unit="sentences",
                    max_sentences=length_controller.max_sentences
                )
            else:  # tokens
                selected = nsga2_select(
                    sentences=sentences,
                    importance=scores,
                    sim_mat=sim_matrix,
                    max_tokens=length_controller.max_tokens,
                    lambda_importance=self.lambda_importance,
                    lambda_coverage=self.lambda_coverage,
                    lambda_redundancy=self.lambda_redundancy,
                    unit="tokens",
                    max_sentences=None
                )
            return selected
        except Exception as e:
            print(f"NSGA-II selection failed: {e}. Falling back to greedy.")
            return self._fallback_greedy(sentences, scores, sim_matrix, length_controller)
    
    def _fallback_greedy(self, sentences, scores, sim_matrix, length_controller):
        greedy_config = {"redundancy": {"lambda": 0.7}}
        greedy = GreedySelector(greedy_config)
        return greedy.select(sentences, scores, sim_matrix, length_controller)

class ThreeStageXLNetSelector:
    """
    三階段選句器：NSGA-II 選 20 句 → XLNet 選 20 句 → MMR 選 3 句
    """
    def __init__(self, config):
        self.config = config
        self.stage_config = config.get("three_stage", {})
        
        # 第一階段：NSGA-II
        self.nsga2_selector = Nsga2SelectorAdapter(config)
        
        # 第二階段：XLNet
        try:
            xlnet_config = self.stage_config.get("xlnet", {})
            model_name = xlnet_config.get("model_name", "xlnet-base-cased")
            self.xlnet_ranker = XLNetRanker(model_name)
            self.use_xlnet = True
            print("XLNet ranker initialized successfully")
        except Exception as e:
            print(f"Warning: XLNet not available: {e}")
            self.use_xlnet = False
        
        # 第三階段：MMR 參數
        self.mmr_lambda = config.get("redundancy", {}).get("lambda", 0.7)
    
    def select(self, sentences, scores, sim_matrix, length_controller):
        """執行三階段選句"""
        print(f"\n=== Three-Stage Selection Process ===")
        print(f"Input: {len(sentences)} sentences")
        
        # 階段 1：NSGA-II 選出 20 句候選
        stage1_candidates = self._stage1_nsga2_selection(sentences, scores, sim_matrix)
        
        # 階段 2：XLNet 獨立選出 20 句候選
        stage2_candidates = self._stage2_xlnet_selection(sentences, scores)
        
        # 階段 3：MMR 從兩個階段的結果中選出最終 3 句
        final_selection = self._stage3_mmr_fusion(sentences, stage1_candidates, stage2_candidates, sim_matrix, length_controller)
        
        print(f"Final selection: {len(final_selection)} sentences")
        return final_selection
    
    def _stage1_nsga2_selection(self, sentences, scores, sim_matrix):
        """階段 1：使用 NSGA-II 選出 20 句候選"""
        target_candidates = self.stage_config.get('stage1_candidates', 20)
        target_candidates = min(target_candidates, len(sentences))
        
        temp_config = {'unit': 'sentences', 'max_sentences': target_candidates}
        temp_controller = LengthController(temp_config, sentences)
        
        try:
            candidates = self.nsga2_selector.select(sentences, scores, sim_matrix, temp_controller)
            print(f"Stage 1 (NSGA-II): Selected {len(candidates)}/{target_candidates} candidates")
            return candidates
        except Exception as e:
            print(f"Stage 1 error: {e}")
            # Fallback: 按分數選擇前 20 句
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return top_indices[:target_candidates]
    
    def _stage2_xlnet_selection(self, sentences, scores):
        """階段 2：使用 XLNet 獨立選出 20 句候選"""
        target_candidates = self.stage_config.get('stage2_candidates', 20)
        target_candidates = min(target_candidates, len(sentences))
        
        if not self.use_xlnet:
            print("Stage 2: XLNet not available, using score-based selection")
            # Fallback: 按分數選擇（與階段1稍有不同的排序）
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return top_indices[:target_candidates]
        
        try:
            # 構建上下文
            context = " ".join(sentences[:min(3, len(sentences))])
            
            # XLNet 對所有句子進行評分和排序
            all_indices = list(range(len(sentences)))
            xlnet_candidates = self.xlnet_ranker.rank_sentences(
                sentences, all_indices, context, target_candidates
            )
            
            print(f"Stage 2 (XLNet): Selected {len(xlnet_candidates)}/{target_candidates} candidates")
            return xlnet_candidates
            
        except Exception as e:
            print(f"Stage 2 error: {e}, using score-based fallback")
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            return top_indices[:target_candidates]
    
    def _stage3_mmr_fusion(self, sentences, stage1_candidates, stage2_candidates, sim_matrix, length_controller):
        """階段 3：使用 MMR 從兩個階段的結果中選出最終句子"""
        
        # 合併兩個階段的候選句子，並去重
        combined_candidates = list(set(stage1_candidates + stage2_candidates))
        
        print(f"Stage 3: MMR fusion from {len(stage1_candidates)} NSGA-II + {len(stage2_candidates)} XLNet = {len(combined_candidates)} unique candidates")
        
        if not combined_candidates:
            return []
        
        max_sentences = length_controller.max_sentences
        selected = []
        remaining = combined_candidates[:]
        
        # 創建候選句子的綜合評分
        candidate_scores = {}
        for idx in combined_candidates:
            # 基礎重要性分數
            base_score = 0.5  # 默認分數
            
            # NSGA-II 候選加分
            if idx in stage1_candidates:
                nsga2_rank = stage1_candidates.index(idx)
                nsga2_score = 1.0 - (nsga2_rank / len(stage1_candidates))
                base_score += 0.5 * nsga2_score
            
            # XLNet 候選加分
            if idx in stage2_candidates:
                xlnet_rank = stage2_candidates.index(idx)
                xlnet_score = 1.0 - (xlnet_rank / len(stage2_candidates))
                base_score += 0.5 * xlnet_score
            
            candidate_scores[idx] = base_score
        
        print(f"Stage 3: MMR selection to {max_sentences} sentences")
        
        # MMR 選擇過程
        while len(selected) < max_sentences and remaining:
            best_idx = None
            best_mmr_score = -float('inf')
            
            for candidate_idx in remaining:
                # 重要性分數（基於綜合評分）
                importance = candidate_scores[candidate_idx]
                
                # 冗餘度計算
                redundancy = 0.0
                if selected and sim_matrix is not None:
                    similarities = [sim_matrix[candidate_idx][sel_idx] for sel_idx in selected]
                    redundancy = max(similarities) if similarities else 0.0
                
                # MMR 分數
                mmr_score = self.mmr_lambda * importance - (1 - self.mmr_lambda) * redundancy
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = candidate_idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                print(f"  Selected sentence {best_idx} (score: {best_mmr_score:.3f})")
            else:
                break
        
        return sorted(selected)

# 測試代碼
if __name__ == "__main__":
    from src.selection.length_controller import LengthController
    
    print("--- Testing ThreeStageXLNetSelector ---")
    
    test_sentences = [
        "The technology company announced record quarterly earnings.",
        "Stock markets showed mixed performance today.",
        "The CEO outlined the company's strategic vision.",
        "Revenue growth exceeded Wall Street expectations.",
        "New product launches are scheduled for next quarter.",
        "The board approved a significant dividend increase.",
        "Market analysts remain optimistic about future growth.",
        "The company invested heavily in artificial intelligence research."
    ]
    
    test_scores = np.array([0.9, 0.6, 0.8, 0.85, 0.7, 0.75, 0.65, 0.8])
    
    # 模擬相似度矩陣
    np.random.seed(42)
    n = len(test_sentences)
    sim_matrix = np.random.rand(n, n)
    sim_matrix = (sim_matrix + sim_matrix.T) / 2
    np.fill_diagonal(sim_matrix, 1.0)
    
    config = {
        "objectives": {"lambda_importance": 1.0, "lambda_coverage": 0.8, "lambda_redundancy": 0.7},
        "nsga2": {"pop_size": 20, "n_gen": 30},
        "three_stage": {
            "stage1_candidates": 6,    # NSGA-II 選 6 句 (測試用)
            "stage2_candidates": 5,    # XLNet 選 5 句 (測試用)
            "xlnet": {"model_name": "xlnet-base-cased"}
        },
        "redundancy": {"lambda": 0.7}
    }
    
    length_controller = LengthController({"unit": "sentences", "max_sentences": 3}, test_sentences)
    
    try:
        selector = ThreeStageXLNetSelector(config)
        selected = selector.select(test_sentences, test_scores, sim_matrix, length_controller)
        
        print(f"\nFinal selection: {selected}")
        print("Selected sentences:")
        for idx in selected:
            print(f"  {idx}: {test_sentences[idx]}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
