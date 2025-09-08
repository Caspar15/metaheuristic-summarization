import numpy as np
from src.selection.length_controller import LengthController

class GreedySelector:
    def __init__(self, config):
        self.redundancy_lambda = config.get("redundancy", {}).get("lambda", 0.7)

    def select(self, sentences, scores, sim_matrix, length_controller):
        num_sentences = len(sentences)
        unselected_indices = list(range(num_sentences))
        
        while not length_controller.is_full():
            best_candidate_idx = -1
            max_mmr_score = -np.inf
            
            selected_so_far = length_controller.get_selected_indices()

            for idx in unselected_indices:
                if idx in selected_so_far:
                    continue

                if length_controller.unit == 'tokens':
                    next_len = len(sentences[idx].split())
                    if length_controller.current_tokens + next_len > length_controller.max_tokens:
                        continue
                
                importance_score = scores[idx]
                redundancy_penalty = 0.0
                if selected_so_far and sim_matrix is not None and sim_matrix.size > 0:
                    sim_with_selected = sim_matrix[idx, selected_so_far]
                    redundancy_penalty = np.max(sim_with_selected)

                mmr_score = (self.redundancy_lambda * importance_score) - \
                            ((1 - self.redundancy_lambda) * redundancy_penalty)
                
                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    best_candidate_idx = idx
            
            if best_candidate_idx == -1:
                break
                
            length_controller.add(best_candidate_idx)

        return length_controller.get_selected_indices()

#testing block
if __name__ == "__main__":

    print("--- Running tests for GreedySelector ---")

    # 1. 準備模擬資料
    test_sentences = [
        "High-scoring and important first sentence.",  # Score: 0.9
        "Second sentence, also very relevant.",        # Score: 0.8
        "This one is almost identical to the second.", # Score: 0.7
        "A different topic, providing diversity.",      # Score: 0.6
        "The final, less important sentence."          # Score: 0.5
    ]
    
    test_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    # 模擬相似度矩陣，讓句子 2 和 1 非常相似
    sim_matrix = np.array([
        [1.0, 0.3, 0.2, 0.1, 0.1],
        [0.3, 1.0, 0.9, 0.4, 0.2], # sent1 vs sent2 = 0.9 (high similarity)
        [0.2, 0.9, 1.0, 0.3, 0.1],
        [0.1, 0.4, 0.3, 1.0, 0.2],
        [0.1, 0.2, 0.1, 0.2, 1.0]
    ])

    # 2. 設定 Config 和 LengthController
    test_config = {
        "redundancy": {"lambda": 0.7},
        "length_control": {"unit": "sentences", "max_sentences": 4}
    }
    
    length_controller = LengthController(test_config["length_control"], test_sentences)

    # 3. 實例化並執行 GreedySelector
    greedy_selector = GreedySelector(test_config)
    selected_indices = greedy_selector.select(
        test_sentences, 
        test_scores, 
        sim_matrix, 
        length_controller
    )

    # 4. 驗證並輸出結果
    print(f"\nConfiguration: Select {test_config['length_control']['max_sentences']} sentences.")
    print(f"MMR lambda (importance weight): {test_config['redundancy']['lambda']}")
    print(f"\nExpected behavior: Should skip sentence 2 (score 0.7) due to high similarity with sentence 1.")
    
    print("\n--- RESULTS ---")
    print(f"Selected indices: {selected_indices}")
    print("Selected sentences:")
    for idx in selected_indices:
        print(f"  - (Score: {test_scores[idx]}) {test_sentences[idx]}")

    assert selected_indices == [0, 1, 3, 4]
    print("\nTest PASSED: Correctly selected 4 sentences and skipped the redundant one.")

