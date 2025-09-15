from typing import List, Tuple, Optional

def count_tokens(text: str) -> int:
    return len(text.split())

def will_fit(current_texts: List[str], candidate: str, max_tokens: int) -> bool:
    total = sum(count_tokens(t) for t in current_texts) + count_tokens(candidate)
    return total <= max_tokens

def will_fit_unit(
    current_texts: List[str],
    candidate: str,
    unit: str = "tokens",
    max_tokens: Optional[int] = None,
    max_sentences: Optional[int] = None,
) -> bool:
    """Check if candidate can be added under the given unit constraint.
    - unit == "tokens": respect max_tokens (fallback if unset).
    - unit == "sentences": cap by number of sentences (count of selected + 1).
    """
    u = (unit or "tokens").lower()
    if u == "sentences":
        if max_sentences is None or max_sentences <= 0:
            return True
        return (len(current_texts) + 1) <= int(max_sentences)
    # default: tokens
    mt = max_tokens if (max_tokens is not None) else 10**9
    return will_fit(current_texts, candidate, int(mt))

def trim_to_max_tokens(texts: List[str], max_tokens: int) -> List[str]:
    out: List[str] = []
    total = 0
    for t in texts:
        ct = count_tokens(t)
        if total + ct <= max_tokens:
            out.append(t)
            total += ct
        else:
            break
    return out

def trim_to_max_sentences(texts: List[str], max_sentences: int) -> List[str]:
    if max_sentences is None or max_sentences <= 0:
        return texts
    return texts[: max_sentences]

# 新增的 LengthController 類
class LengthController:
    """
    長度控制器類，用於統一管理句子和token的長度限制
    """
    def __init__(self, config: dict, sentences: List[str]):
        self.unit = config.get('unit', 'tokens')
        self.max_tokens = config.get('max_tokens', 100)
        self.max_sentences = config.get('max_sentences', 3)
        self.sentences = sentences

    def will_fit_unit(self, current_texts: List[str], candidate: str) -> bool:
        """檢查候選句子是否可以在當前限制下添加"""
        u = (self.unit or "tokens").lower()
        if u == "sentences":
            if self.max_sentences is None or self.max_sentences <= 0:
                return True
            return (len(current_texts) + 1) <= int(self.max_sentences)
        # default: tokens
        mt = self.max_tokens if (self.max_tokens is not None) else 10**9
        return self.will_fit(current_texts, candidate, int(mt))

    @staticmethod
    def count_tokens(text: str) -> int:
        """計算文本中的token數量"""
        return len(text.split())

    def will_fit(self, current_texts: List[str], candidate: str, max_tokens: int) -> bool:
        """檢查是否可以在token限制內添加候選句子"""
        total = sum(self.count_tokens(t) for t in current_texts) + self.count_tokens(candidate)
        return total <= max_tokens

    def trim_to_max_tokens(self, texts: List[str]) -> List[str]:
        """裁剪文本列表以符合最大token限制"""
        out: List[str] = []
        total = 0
        for t in texts:
            ct = self.count_tokens(t)
            if total + ct <= self.max_tokens:
                out.append(t)
                total += ct
            else:
                break
        return out

    def trim_to_max_sentences(self, texts: List[str]) -> List[str]:
        """裁剪文本列表以符合最大句子數限制"""
        if self.max_sentences is None or self.max_sentences <= 0:
            return texts
        return texts[:self.max_sentences]

# 測試代碼
if __name__ == "__main__":
    sentences = [
        "The cat sits on the mat.",
        "Dogs are loyal animals.",
        "Artificial intelligence is transforming the world.",
        "The quick brown fox jumps over the lazy dog.",
        "Data science is an interdisciplinary field.",
    ]
    
    # 測試基於token的控制
    config_tokens = {"unit": "tokens", "max_tokens": 15}
    lc_token = LengthController(config_tokens, sentences)
    print(f"Token mode - can fit first sentence: {lc_token.will_fit_unit([], sentences[0])}")
    
    # 測試基於句子數的控制
    config_sentences = {"unit": "sentences", "max_sentences": 3}
    lc_sent = LengthController(config_sentences, sentences)
    print(f"Sentence mode - can fit 3rd sentence with 2 existing: {lc_sent.will_fit_unit(sentences[:2], sentences[2])}")
    print(f"Sentence mode - can fit 4th sentence with 3 existing: {lc_sent.will_fit_unit(sentences[:3], sentences[4])}")
    
    print("LengthController 測試完成")
