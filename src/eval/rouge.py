from typing import List, Dict


def rouge_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except Exception as e:
        raise RuntimeError("請先安裝 rouge-score 以計算 ROUGE") from e

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = max(1, len(preds))
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        totals["rouge1"] += s["rouge1"].fmeasure
        totals["rouge2"] += s["rouge2"].fmeasure
        totals["rougeL"] += s["rougeL"].fmeasure
    return {k: v / n for k, v in totals.items()}

