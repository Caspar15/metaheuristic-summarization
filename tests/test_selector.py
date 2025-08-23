from __future__ import annotations

from src.models.extractive.greedy import select_greedy


def test_redundancy_removal_prefers_one_sentence():
    sents = [
        {"text": "Apple releases new iPhone today", "tokens": ["apple", "releases", "new", "iphone", "today"]},
        {"text": "New iPhone is released by Apple", "tokens": ["new", "iphone", "is", "released", "by", "apple"]},
    ]
    scores = [1.0, 0.9]
    idxs, summary = select_greedy(
        sentences=sents,
        scores=scores,
        max_tokens=100,
        redundancy_ngram=2,
        redundancy_threshold=0.5,
    )
    assert len(idxs) == 1
    assert summary.strip() != ""

