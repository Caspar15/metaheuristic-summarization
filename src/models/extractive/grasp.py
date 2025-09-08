# src/models/extractive/grasp.py

import random
import numpy as np
from src.selection.length_controller import LengthController

class GraspSelector:
    """
    GRASP (Greedy Randomized Adaptive Search Procedure) 選句演算法
    """
    def __init__(self, config):
        self.redundancy_lambda = config.get("redundancy", {}).get("lambda", 0.7)
        self.iterations = config.get("grasp", {}).get("iterations", 10)
        self.rcl_alpha = config.get("grasp", {}).get("rcl_alpha", 0.3)
        self.seed = config.get("seed", 2024)

    def select(self, sentences, scores, sim_matrix, length_controller):
        """
        執行 GRASP 選句，由傳入的 length_controller 控制長度。
        """
        best_solution = []
        best_score = -float('inf')
        
        for iteration in range(self.iterations):
            # 每次迭代都需要一個新的 controller (因為它有內部狀態)
            iter_controller = LengthController(
                {
                    'unit': length_controller.unit,
                    'max_tokens': length_controller.max_tokens,
                    'max_sentences': length_controller.max_sentences
                },
                sentences
            )
            
            # 建構階段：使用 RCL 隨機貪婪建構
            solution = self._construct_solution(sentences, scores, sim_matrix, iter_controller)
            
            # 局部搜尋階段：swap 操作（維持句子數量不變）
            solution = self._local_search(solution, sentences, scores, sim_matrix)
            
            # 評估解的品質
            solution_score = self._evaluate_solution(solution, scores, sim_matrix)
            
            if solution_score > best_score:
                best_score = solution_score
                best_solution = solution[:]
        
        return sorted(best_solution)

    def _construct_solution(self, sentences, scores, sim_matrix, length_controller):
        """
        GRASP 建構階段：RCL (Restricted Candidate List) 隨機選擇
        """
        num_sentences = len(sentences)
        unselected = list(range(num_sentences))
        
        while not length_controller.is_full() and unselected:
            selected_so_far = length_controller.get_selected_indices()
            
            # 計算所有未選句子的 MMR 分數
            mmr_scores = []
            valid_candidates = []
            
            for idx in unselected:
                if idx in selected_so_far:
                    continue
                    
                # 預先檢查長度約束
                if length_controller.unit == 'tokens':
                    next_len = len(sentences[idx].split())
                    if length_controller.current_tokens + next_len > length_controller.max_tokens:
                        continue
                
                importance_score = scores[idx]
                redundancy_penalty = 0.0
                
                if selected_so_far and sim_matrix is not None:
                    sim_with_selected = sim_matrix[idx, selected_so_far]
                    redundancy_penalty = np.max(sim_with_selected)
                
                mmr_score = (self.redundancy_lambda * importance_score) - \
                           ((1 - self.redundancy_lambda) * redundancy_penalty)
                
                mmr_scores.append(mmr_score)
                valid_candidates.append(idx)
            
            if not valid_candidates:
                break
            
            # 建立 RCL：選擇分數在 [max - α(max-min), max] 範圍內的候選
            max_score = max(mmr_scores)
            min_score = min(mmr_scores)
            threshold = max_score - self.rcl_alpha * (max_score - min_score)
            
            rcl = [valid_candidates[i] for i, score in enumerate(mmr_scores) 
                   if score >= threshold]
            
            # 隨機選擇一個候選
            chosen_idx = random.choice(rcl)
            length_controller.add(chosen_idx)
            unselected.remove(chosen_idx)
        
        return length_controller.get_selected_indices()

    def _local_search(self, solution, sentences, scores, sim_matrix):
        """
        局部搜尋：嘗試 swap 操作（交換已選和未選的句子）
        """
        if not solution:
            return solution
            
        improved = True
        current_solution = solution[:]
        current_score = self._evaluate_solution(current_solution, scores, sim_matrix)
        
        while improved:
            improved = False
            unselected = [i for i in range(len(sentences)) if i not in current_solution]
            
            for selected_idx in current_solution[:]:
                for unselected_idx in unselected:
                    # 嘗試交換
                    new_solution = current_solution[:]
                    new_solution.remove(selected_idx)
                    new_solution.append(unselected_idx)
                    
                    new_score = self._evaluate_solution(new_solution, scores, sim_matrix)
                    
                    if new_score > current_score:
                        current_solution = new_solution
                        current_score = new_score
                        improved = True
                        break
                
                if improved:
                    break
        
        return sorted(current_solution)

    def _evaluate_solution(self, solution, scores, sim_matrix):
        """
        評估解的品質（重要性 - 冗餘度）
        """
        if not solution:
            return 0.0
        
        importance = sum(scores[i] for i in solution)
        redundancy = 0.0
        
        if len(solution) > 1 and sim_matrix is not None:
            for i in range(len(solution)):
                for j in range(i + 1, len(solution)):
                    redundancy += sim_matrix[solution[i], solution[j]]
            redundancy /= (len(solution) * (len(solution) - 1) / 2)
        
        return self.redundancy_lambda * importance - (1 - self.redundancy_lambda) * redundancy


# testing block
if __name__ == "__main__":
    import sys
    import os
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    sys.path.insert(0, project_root)
    
    from src.selection.length_controller import LengthController
    
    print("--- Testing GraspSelector ---")
    
    test_sentences = [
        "First important sentence.",
        "Second relevant sentence.",
        "Third sentence with different topic.",
        "Fourth sentence somewhat similar to first.",
        "Fifth sentence with unique content."
    ]
    
    test_scores = np.array([0.9, 0.8, 0.6, 0.7, 0.5])
    sim_matrix = np.array([
        [1.0, 0.3, 0.1, 0.8, 0.2],
        [0.3, 1.0, 0.2, 0.4, 0.1],
        [0.1, 0.2, 1.0, 0.1, 0.3],
        [0.8, 0.4, 0.1, 1.0, 0.2],
        [0.2, 0.1, 0.3, 0.2, 1.0]
    ])
    
    config = {
        "redundancy": {"lambda": 0.7},
        "grasp": {"iterations": 5, "rcl_alpha": 0.3},
        "seed": 42
    }
    
    length_controller = LengthController(
        {"unit": "sentences", "max_sentences": 3}, 
        test_sentences
    )
    
    grasp_selector = GraspSelector(config)
    selected = grasp_selector.select(test_sentences, test_scores, sim_matrix, length_controller)
    
    print(f"Selected indices: {selected}")
    print("Selected sentences:")
    for idx in selected:
        print(f"  - {test_sentences[idx]}")
    
    assert len(selected) == 3
    print("Test PASSED!")
