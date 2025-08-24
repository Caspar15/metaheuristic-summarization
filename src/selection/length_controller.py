from typing import List, Tuple


def count_tokens(text: str) -> int:
    return len(text.split())


def will_fit(current_texts: List[str], candidate: str, max_tokens: int) -> bool:
    total = sum(count_tokens(t) for t in current_texts) + count_tokens(candidate)
    return total <= max_tokens


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

