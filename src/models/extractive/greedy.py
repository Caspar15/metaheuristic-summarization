from __future__ import annotations

from typing import Dict, List, Tuple


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    if n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def select_greedy(
    sentences: List[Dict[str, object]],
    scores: List[float],
    max_tokens: int = 100,
    redundancy_ngram: int = 3,
    redundancy_threshold: float = 0.6,
) -> Tuple[List[int], str]:
    """Greedy selection by descending score with simple redundancy removal.

    Returns selected indices and the concatenated summary string.
    """
    # Prepare items with index to keep stable references
    items = [
        {
            "idx": i,
            "text": sentences[i].get("text", ""),
            "tokens": list(sentences[i].get("tokens", [])),
            "score": float(scores[i]),
        }
        for i in range(len(sentences))
    ]

    # Sort by score descending
    items.sort(key=lambda x: x["score"], reverse=True)

    selected: List[Dict[str, object]] = []
    selected_ng: set = set()
    token_budget = 0

    for it in items:
        toks = list(it["tokens"])  # type: ignore[call-arg]
        cand_ng = set(_ngrams(toks, redundancy_ngram)) if redundancy_ngram > 0 else set()
        sim = _jaccard(cand_ng, selected_ng) if cand_ng else 0.0
        if sim > redundancy_threshold:
            continue
        if token_budget + len(toks) > max_tokens:
            continue
        selected.append(it)
        token_budget += len(toks)
        selected_ng |= cand_ng

    # Keep original document order for readability
    selected.sort(key=lambda x: x["idx"])
    summary = " ".join(s["text"] for s in selected)
    indices = [s["idx"] for s in selected]
    return indices, summary

