from __future__ import annotations

from src.eval.rouge import compute_rouge


def test_compute_rouge_has_keys():
    sys = "apple releases phone"
    ref = "apple releases new phone"
    scores = compute_rouge(system=sys, reference=ref, types=["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    assert set(["rouge1", "rouge2", "rougeL"]).issubset(scores.keys())

