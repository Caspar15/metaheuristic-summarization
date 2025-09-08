import argparse
import os
import json
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

from src.utils.io import (
    load_yaml,
    ensure_dir,
    now_stamp,
    read_jsonl,
    write_jsonl,
    set_global_seed,
)
from src.features.tf_isf import sentence_tf_isf_scores
from src.features.length import length_scores
from src.features.position import position_scores
from src.features.compose import combine_scores
from src.representations.sent_vectors import SentenceVectors
from src.representations.similarity import cosine_similarity_matrix
from src.selection.candidate_pool import topk_by_score
from src.models.extractive.greedy import GreedySelector
from src.selection.length_controller import LengthController


def build_base_scores(sentences: List[str], cfg: Dict) -> List[float]:
    f_importance = sentence_tf_isf_scores(sentences)
    f_len = length_scores(sentences)
    f_pos = position_scores(sentences)
    weights = {
        "importance": float(cfg.get("objectives", {}).get("lambda_importance", 1.0)),
        "length": 0.3,
        "position": 0.3,
    }
    feats = {"importance": f_importance, "length": f_len, "position": f_pos}
    return combine_scores(feats, weights)

def _topk_by_position(sentences: List[str], k: int) -> List[int]:
    pos = position_scores(sentences)
    idx = sorted(range(len(pos)), key=lambda i: pos[i], reverse=True)
    return idx[:k]

def _topk_by_centrality_tfidf(sentences: List[str], k: int) -> List[int]:
    if len(sentences) == 0:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    X = TfidfVectorizer(lowercase=True).fit_transform(sentences)
    sim = cosine_similarity(X)
    cent = sim.mean(axis=1)
    idx = sorted(range(len(sentences)), key=lambda i: float(cent[i]), reverse=True)
    return idx[:k]

def _build_candidate_union(
    sentences: List[str], base_scores: List[float], k: int, sources: List[str]
) -> List[int]:
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
    if not cand:
        cand.update(topk_by_score(base_scores, k))
    return sorted(cand)

def _greedy_oracle_indices(sentences: List[str], reference: str, max_tokens: int) -> List[int]:
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
    def _count_tokens(s: str) -> int:
        return len((s or "").split())
    for _ in range(n):
        best_i = None
        best_gain = 0.0
        for i in range(n):
            if i in selected:
                continue
            t = _count_tokens(sentences[i])
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
        cur_tokens += _count_tokens(sentences[best_i])
        best_f = scorer.score(reference or "", cur_summary)["rouge1"].fmeasure
    return sorted(selected)

def summarize_one(doc: Dict, cfg: Dict, selector) -> Dict:
    sentences: List[str] = doc.get("sentences", [])
    highlights: str = doc.get("highlights", "")

    base_scores = build_base_scores(sentences, cfg)
    rep_cfg = cfg.get("representations", {})
    sim = None
    if bool(rep_cfg.get("use", True)) and len(sentences) > 0:
        vec = SentenceVectors(method=rep_cfg.get("method", "tfidf"))
        X = vec.fit_transform(sentences)
        sim = cosine_similarity_matrix(X)

    cand_cfg = cfg.get("candidates", {})
    k = int(cand_cfg.get("k", min(15, len(sentences))))
    use_cand = bool(cand_cfg.get("use", True))
    mode = (cand_cfg.get("mode", "hard") or "hard").lower()
    sources = cand_cfg.get("sources", ["score"]) or ["score"]
    soft_boost = float(cand_cfg.get("soft_boost", 1.05))
    cand_idx = _build_candidate_union(sentences, base_scores, k, sources)

    recall_target = cand_cfg.get("recall_target", None)
    if recall_target is not None and isinstance(recall_target, (int, float)) and 0 < recall_target <= 1:
        oracle = _greedy_oracle_indices(sentences, doc.get("highlights", ""), int(cfg.get("length_control", {}).get("max_tokens", 100)))
        if oracle:
            import math as _m
            used_k = max(1, len(cand_idx))
            while used_k < len(sentences):
                inter = len(set(cand_idx) & set(oracle))
                rec = inter / max(1, len(oracle))
                if rec >= float(recall_target) - 1e-12:
                    break
                used_k = min(len(sentences), max(used_k + 1, int(_m.ceil(used_k * 1.5))))
                cand_idx = _build_candidate_union(sentences, base_scores, used_k, sources)

    if use_cand and cand_idx:
        if mode == "hard":
            sub_sentences = [sentences[i] for i in cand_idx]
            sub_scores = [base_scores[i] for i in cand_idx]
            sub_sim = sim[np.ix_(cand_idx, cand_idx)] if sim is not None else None
        else:  # soft mode
            sub_sentences = sentences
            sub_scores = base_scores[:]
            for i in cand_idx:
                sub_scores[i] *= soft_boost
            sub_sim = sim
    else:
        sub_sentences, sub_scores, sub_sim = sentences, base_scores, sim

    length_ctrl_config = cfg.get('length_control', {})
    length_controller = LengthController(length_ctrl_config, sub_sentences)

    picked_sub = selector.select(sub_sentences, sub_scores, sub_sim, length_controller)

    if use_cand and cand_idx and mode == "hard":
        selected = sorted([cand_idx[i] for i in picked_sub])
    else:
        selected = sorted(picked_sub)

    summary = " ".join([sentences[i] for i in selected])
    return {
        "id": doc.get("id"),
        "selected_indices": selected,
        "summary": summary,
        "reference": highlights,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config yaml")
    ap.add_argument("--split", required=True, help="dataset split name")
    ap.add_argument("--input", required=True, help="processed jsonl path")
    ap.add_argument("--run_dir", default="runs", help="runs output root")
    ap.add_argument("--stamp", default=None, help="optional fixed stamp for output dir")
    ap.add_argument("--optimizer", default=None, help="override optimizer.method in config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.optimizer:
        cfg.setdefault("optimizer", {})["method"] = args.optimizer
    set_global_seed(cfg.get("seed"))
    stamp = args.stamp or now_stamp()
    out_dir = os.path.join(args.run_dir, stamp)
    ensure_dir(out_dir)

    optimizer_name = cfg.get("optimizer", {}).get("method", "greedy").lower()
    
    if optimizer_name == "greedy":
        selector = GreedySelector(cfg)
        print(f"Using GreedySelector")
        
    elif optimizer_name == "grasp":
        try:
            from src.models.extractive.grasp import GraspSelector
            selector = GraspSelector(cfg)
            print(f"Using GraspSelector")
        except ImportError as e:
            print(f"Warning: Could not import GraspSelector: {e}. Falling back to GreedySelector.")
            selector = GreedySelector(cfg)
            
    elif optimizer_name == "nsga2":
        try:
            from src.models.extractive.nsga2 import Nsga2Selector
            selector = Nsga2Selector(cfg)
            print(f"Using Nsga2Selector")
        except ImportError as e:
            print(f"Warning: Could not import Nsga2Selector: {e}. Falling back to GreedySelector.")
            selector = GreedySelector(cfg)
            
    else:
        print(f"Warning: Unknown optimizer '{optimizer_name}'. Using GreedySelector.")
        selector = GreedySelector(cfg)
    
    preds_path = os.path.join(out_dir, "predictions.jsonl")
    rows = []
    
    for doc in tqdm(read_jsonl(args.input), desc=f"Selecting with {optimizer_name}"):
        rows.append(summarize_one(doc, cfg, selector))
    
    write_jsonl(preds_path, rows)

    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
