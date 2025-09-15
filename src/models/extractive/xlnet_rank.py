import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import numpy as np
from typing import List, Tuple
import logging

class XLNetRanker:
    """
    使用 XLNet 對句子進行重要性評分和排序
    """
    def __init__(self, model_name="xlnet-base-cased", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        try:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
            self.model = XLNetForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=1
            )
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"XLNet model loaded on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load XLNet model: {e}")
            raise
    
    def score_sentences(self, sentences: List[str], context: str = "") -> List[float]:
        """
        對句子列表進行重要性評分
        """
        scores = []
        
        with torch.no_grad():
            for sentence in sentences:
                # 構建輸入文本
                if context:
                    input_text = f"{context} [SEP] {sentence}"
                else:
                    input_text = sentence
                
                # 分詞和編碼
                inputs = self.tokenizer(
                    input_text,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 前向傳播
                outputs = self.model(**inputs)
                
                # 獲取重要性分數
                logits = outputs.logits
                score = torch.sigmoid(logits).cpu().item()
                scores.append(score)
        
        return scores
    
    def rank_sentences(self, sentences: List[str], indices: List[int], 
                      context: str = "", top_k: int = 20) -> List[int]:
        """
        對指定索引的句子進行排序
        
        Args:
            sentences: 完整句子列表
            indices: 需要評分的句子索引
            context: 上下文
            top_k: 返回前k個句子的索引
            
        Returns:
            排序後的句子索引列表
        """
        if not indices:
            return []
        
        # 提取候選句子
        candidate_sentences = [sentences[i] for i in indices]
        
        # XLNet 評分
        scores = self.score_sentences(candidate_sentences, context)
        
        # 創建 (index, score) 對並排序
        indexed_scores = [(indices[i], score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回排序後的索引
        return [idx for idx, _ in indexed_scores[:top_k]]


class XLNetSelector:
    """
    基於 XLNet 的選句器（單階段使用）
    """
    def __init__(self, config):
        self.config = config
        xlnet_config = config.get("xlnet", {})
        model_name = xlnet_config.get("model_name", "xlnet-base-cased")
        self.ranker = XLNetRanker(model_name)
    
    def select(self, sentences, scores, sim_matrix, length_controller):
        """
        使用 XLNet 進行選句
        """
        if not sentences:
            return []
        
        # 構建上下文（使用前3句）
        context = " ".join(sentences[:3])
        
        # 所有句子的索引
        all_indices = list(range(len(sentences)))
        
        # XLNet 排序
        ranked_indices = self.ranker.rank_sentences(
            sentences, all_indices, context, 
            top_k=length_controller.max_sentences
        )
        
        return sorted(ranked_indices[:length_controller.max_sentences])


# 測試代碼
if __name__ == "__main__":
    print("--- Testing XLNetRanker ---")
    
    test_sentences = [
        "The company reported strong quarterly earnings.",
        "Stock prices fell amid market uncertainty.",
        "The CEO announced new strategic initiatives.",
        "Revenue growth exceeded analyst expectations.",
        "The board approved a dividend increase."
    ]
    
    config = {
        "xlnet": {
            "model_name": "xlnet-base-cased"
        }
    }
    
    try:
        selector = XLNetSelector(config)
        
        # 模擬 length_controller
        class MockLengthController:
            max_sentences = 3
        
        selected = selector.select(test_sentences, [], None, MockLengthController())
        
        print(f"Selected indices: {selected}")
        print("Selected sentences:")
        for idx in selected:
            print(f"  {idx}: {test_sentences[idx]}")
            
    except Exception as e:
        print(f"Test failed: {e}")
