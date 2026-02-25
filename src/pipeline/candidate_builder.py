"""Candidate pool construction logic extracted from select_sentences.py."""

from typing import Dict, List, Optional, Set

import numpy as np

from src.utils.tokenizer import count_tokens
from src.features.position import position_scores
from src.features.graph import compute_textrank_scores
from src.representations.tfidf_helper import tfidf_centroid_ranks, tfidf_scores_and_sim
from src.selection.candidate_pool import topk_by_score


def _topk_by_position(sentences: List[str], k: int) -> List[int]:
    pos = position_scores(sentences)
    idx = sorted(range(len(pos)), key=lambda i: pos[i], reverse=True)
    return idx[:k]


def _topk_by_centrality_tfidf(sentences: List[str], k: int) -> List[int]:
    return tfidf_centroid_ranks(sentences, k)


def _topk_by_graph_score(
    sentences: List[str], k: int, sim_matrix=None, threshold: float = 0.0
) -> List[int]:
    if not sentences:
        return []
    scores = []
    if sim_matrix is not None:
        scores = compute_textrank_scores(sim_matrix, threshold=threshold)
    else:
        _, sim = tfidf_scores_and_sim(sentences, sublinear_tf=False, ngram_range=(1, 1))
        scores = compute_textrank_scores(sim)
    idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    return idx[:k]


def build_candidate_union(
    sentences: List[str],
    base_scores: List[float],
    k: int,
    sources: List[str],
    sim_matrix=None,
    threshold: float = 0.0,
) -> List[int]:
    """Build candidate pool from multiple sources as a union of top-k sets."""
    n = len(sentences)
    if n == 0:
        return []
    k = min(max(1, k), n)
    cand: Set[int] = set()
    for src in sources:
        name = (src or "").strip().lower()
        if name == "score":
            cand.update(topk_by_score(base_scores, k))
        elif name == "position":
            cand.update(_topk_by_position(sentences, k))
        elif name == "centrality":
            try:
                cand.update(_topk_by_centrality_tfidf(sentences, k))
            except Exception:
                pass
        elif name in ("graph", "textrank"):
            try:
                cand.update(_topk_by_graph_score(sentences, k, sim_matrix, threshold=threshold))
            except Exception:
                pass
    if not cand:
        cand.update(topk_by_score(base_scores, k))
    return sorted(cand)


def greedy_oracle_indices(
    sentences: List[str], reference: str, max_tokens: int
) -> List[int]:
    """Greedy oracle by ROUGE-1 F gain (used for recall_target of candidates)."""
    try:
        from rouge_score import rouge_scorer
    except Exception:
        return []
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    selected: List[int] = []
    cur_summary = ""
    cur_tokens = 0
    best_f = 0.0
    n = len(sentences)
    for _ in range(n):
        best_i = None
        best_gain = 0.0
        for i in range(n):
            if i in selected:
                continue
            t = count_tokens(sentences[i])
            if cur_tokens + t > max_tokens:
                continue
            cand = (cur_summary + " " + sentences[i]).strip()
            f = scorer.score(reference or "", cand)["rouge1"].fmeasure
            gain = f - best_f
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        cur_summary = (cur_summary + " " + sentences[best_i]).strip()
        cur_tokens += count_tokens(sentences[best_i])
        best_f = scorer.score(reference or "", cur_summary)["rouge1"].fmeasure
    return sorted(selected)
