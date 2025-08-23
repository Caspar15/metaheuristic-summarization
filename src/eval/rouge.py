from __future__ import annotations

from typing import Dict, List

from rouge_score import rouge_scorer


def compute_rouge(
    system: str,
    reference: str,
    types: List[str],
    use_stemmer: bool = True,
) -> Dict[str, float]:
    """Compute ROUGE F1 scores for the given types using rouge-score."""
    scorer = rouge_scorer.RougeScorer(types, use_stemmer=use_stemmer)
    scores = scorer.score(reference, system)
    # Return only F1 scores as floats
    return {k: float(v.fmeasure) for k, v in scores.items()}

