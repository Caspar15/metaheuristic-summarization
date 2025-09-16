from typing import List, Optional


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

