# src/selection/length_controller.py

class LengthController:
    """
    根據設定檔中的單位（tokens 或 sentences）來管理摘要的長度約束。
    """
    def __init__(self, config, sentences):
        """
        初始化長度控制器。

        Args:
            config (dict): length_control 區塊的設定。
            sentences (list[str]): 文章的所有句子列表，用於計算 token 數。
        """
        self.unit = config.get("unit", "tokens")
        self.max_tokens = config.get("max_tokens", 100)
        self.max_sentences = config.get("max_sentences")
        self.sentences = sentences
        
        self.current_tokens = 0
        self.current_sentences = 0
        self.selected_indices = []

    def add(self, sentence_index):
        """
        將一個句子加入到摘要中，並更新內部狀態。

        Args:
            sentence_index (int): 要加入的句子的索引。
        """
        if sentence_index in self.selected_indices:
            return  # 防止重複加入

        self.selected_indices.append(sentence_index)
        self.current_sentences += 1
        
        sentence_text = self.sentences[sentence_index]
        self.current_tokens += len(sentence_text.split())

    def is_full(self):
        """
        檢查是否已達到長度上限。

        Returns:
            bool: 如果已滿則返回 True，否則返回 False。
        """
        if self.unit == "sentences":
            if self.max_sentences is None:
                # 如果 max_sentences 未設定，為避免無限迴圈，給予警告並退回 token 限制
                print("Warning: length_control.unit is 'sentences' but max_sentences is not set. Falling back to max_tokens.")
                return self.current_tokens >= self.max_tokens
            return self.current_sentences >= self.max_sentences
        
        elif self.unit == "tokens":
            return self.current_tokens >= self.max_tokens
            
        else:
            # 不支援的單位，預設為 False 並給予警告
            print(f"Warning: Unknown length_control.unit '{self.unit}'. Length control is disabled.")
            return False

    def get_selected_indices(self):
        """
        返回已選句子的索引列表。
        """
        return self.selected_indices

# testing block
if __name__ == "__main__":
    print("--- Running tests for LengthController ---")
    
    test_sentences = [
        "This is the first sentence.",         # 5 tokens
        "Here is the second one.",             # 5 tokens
        "And a third.",                        # 3 tokens
        "The fourth sentence appears now.",    # 5 tokens
        "Finally, the fifth one is this."      # 6 tokens
    ]

    # --- 測試案例 1: 按句子數量控制 (unit: "sentences") ---
    print("\n[Test Case 1: Control by number of sentences]")
    config_by_sentence = {
        "unit": "sentences",
        "max_sentences": 3
    }
    lc_sent = LengthController(config_by_sentence, test_sentences)
    
    print(f"Configuration: unit='{lc_sent.unit}', max_sentences={lc_sent.max_sentences}")
    for i in range(len(test_sentences)):
        if lc_sent.is_full():
            print(f"-> Controller is full. Stopping.")
            break
        print(f"Adding sentence {i}...")
        lc_sent.add(i)
        
    print(f"Final selected indices: {lc_sent.get_selected_indices()}")
    print(f"Result: Selected {lc_sent.current_sentences} sentences, {lc_sent.current_tokens} tokens.")
    assert lc_sent.get_selected_indices() == [0, 1, 2]
    print("Test Case 1: PASSED")

    # --- 測試案例 2: 按 token 數量控制 (unit: "tokens") ---
    print("\n[Test Case 2: Control by number of tokens]")
    config_by_token = {
        "unit": "tokens",
        "max_tokens": 12
    }
    lc_token = LengthController(config_by_token, test_sentences)

    print(f"Configuration: unit='{lc_token.unit}', max_tokens={lc_token.max_tokens}")
    for i in range(len(test_sentences)):
        # **修正點**：在加入前，先模擬加入後的 token 總數來判斷是否會超長
        next_sentence_len = len(test_sentences[i].split())
        if lc_token.current_tokens + next_sentence_len > lc_token.max_tokens:
             print(f"-> Adding sentence {i} would exceed max_tokens. Stopping.")
             break

        print(f"Adding sentence {i} (tokens: {next_sentence_len})...")
        lc_token.add(i)

    print(f"Final selected indices: {lc_token.get_selected_indices()}")
    print(f"Result: Selected {lc_token.current_sentences} sentences, {lc_token.current_tokens} tokens.")
    assert lc_token.get_selected_indices() == [0, 1]
    print("Test Case 2: PASSED")
    
    print("\n--- All tests completed successfully! ---")
