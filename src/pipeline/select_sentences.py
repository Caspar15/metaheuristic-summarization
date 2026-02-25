"""Main sentence-selection pipeline.

Orchestrates feature building, candidate-pool construction, and
optimizer dispatch — each delegated to its own module.
"""

import argparse
import os
import time
from typing import Dict, List

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
from src.representations.sent_vectors import SentenceVectors
from src.representations.similarity import cosine_similarity_matrix

from src.pipeline.feature_builder import build_base_scores
from src.pipeline.candidate_builder import build_candidate_union, greedy_oracle_indices
from src.pipeline.optimizer_dispatch import dispatch_optimizer


# ------------------------------------------------------------------ #
#  Core per-document summarisation                                     #
# ------------------------------------------------------------------ #

def summarize_one(doc: Dict, cfg: Dict) -> Dict:
    sentences: List[str] = doc.get("sentences", [])
    highlights: str = doc.get("highlights", "")

    # 1. Similarity matrix (needed by graph features, candidates, NSGA-II)
    rep_cfg = cfg.get("representations", {})
    sim = None
    if bool(rep_cfg.get("use", True)) and len(sentences) > 0:
        method = rep_cfg.get("method", "tfidf")
        vec = SentenceVectors(method=method)
        X = vec.fit_transform(sentences)
        sim = cosine_similarity_matrix(X)

    # 2. Feature scores
    base_scores = build_base_scores(sentences, cfg, similarity_matrix=sim)

    # 3. Length / redundancy parameters
    lc = cfg.get("length_control", {})
    unit = (lc.get("unit", "tokens") or "tokens").lower()
    max_tokens = int(lc.get("max_tokens", 100))
    max_sents_limit = lc.get("max_sentences", None)
    max_sents = int(max_sents_limit) if (max_sents_limit is not None) else None
    alpha = float(cfg.get("redundancy", {}).get("lambda", 0.7))

    # 4. Candidate pool
    cand_cfg = cfg.get("candidates", {})
    k = int(cand_cfg.get("k", min(15, len(sentences))))
    use_cand = bool(cand_cfg.get("use", True))
    mode = (cand_cfg.get("mode", "hard") or "hard").lower()
    sources = cand_cfg.get("sources", ["score"]) or ["score"]
    soft_boost = float(cand_cfg.get("soft_boost", 1.05))

    g_thresh = float(cfg.get("graph_params", {}).get("threshold", 0.0))
    cand_idx = build_candidate_union(sentences, base_scores, k, sources, sim_matrix=sim, threshold=g_thresh)

    # Optional dynamic k to reach recall target
    recall_target = cand_cfg.get("recall_target", None)
    if recall_target is not None and isinstance(recall_target, (int, float)) and 0 < recall_target <= 1:
        oracle = greedy_oracle_indices(sentences, doc.get("highlights", ""), int(cfg.get("length_control", {}).get("max_tokens", 100)))
        if oracle:
            import math as _m
            used_k = max(1, len(cand_idx))
            while used_k < len(sentences):
                inter = len(set(cand_idx) & set(oracle))
                rec = inter / max(1, len(oracle))
                if rec >= float(recall_target) - 1e-12:
                    break
                used_k = min(len(sentences), max(used_k + 1, int(_m.ceil(used_k * 1.5))))
                cand_idx = build_candidate_union(sentences, base_scores, used_k, sources, sim_matrix=sim)

    # 5. Apply candidate mode
    if use_cand and cand_idx:
        if mode == "hard":
            sub_sentences = [sentences[i] for i in cand_idx]
            sub_scores = [base_scores[i] for i in cand_idx]
            sub_sim = sim[np.ix_(cand_idx, cand_idx)] if sim is not None else None
        else:
            sub_sentences = sentences
            sub_scores = base_scores[:]
            for i in cand_idx:
                sub_scores[i] = float(sub_scores[i]) * soft_boost
            sub_sim = sim
    else:
        sub_sentences = sentences
        sub_scores = base_scores
        sub_sim = sim

    # 6. Optimizer dispatch
    if unit == "words":
        # Simple word-budget greedy (no optimizer)
        max_words = int(lc.get("max_words", 400))
        picked_sub = []
        current_words = 0
        sorted_indices = np.argsort(sub_scores)[::-1]
        for idx in sorted_indices:
            sentence_text = sub_sentences[idx]
            word_count = len(sentence_text.split())
            if current_words + word_count <= max_words:
                picked_sub.append(idx)
                current_words += word_count
    else:
        method_opt = cfg.get("optimizer", {}).get("method", "greedy").lower()
        picked_sub = dispatch_optimizer(
            method_opt, sub_sentences, sub_scores, sub_sim,
            max_tokens, cfg, alpha, unit, max_sents,
        )

    # 7. Map back to original indices
    if use_cand and cand_idx and mode == "hard":
        selected = sorted(cand_idx[i] for i in picked_sub)
    else:
        selected = sorted(picked_sub)

    selected.sort()
    summary = " ".join([sentences[i] for i in selected])
    return {
        "id": doc.get("id"),
        "selected_indices": selected,
        "summary": summary,
        "reference": highlights,
    }


# ------------------------------------------------------------------ #
#  CLI entry-point                                                     #
# ------------------------------------------------------------------ #

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
        cfg.setdefault("optimizer", {})
        cfg["optimizer"]["method"] = args.optimizer

    # Guard: Stage2 union input should use fast (non-BERT) optimizers only
    method_opt = (cfg.get("optimizer", {}).get("method") or "").lower()
    in_path = str(args.input)
    is_stage2_union = ("stage2" in in_path and "union" in in_path)
    if is_stage2_union and method_opt in ("bert", "roberta", "xlnet", "fused"):
        raise RuntimeError(
            f"Stage2 union input detected ({in_path}). Please use non-BERT optimizers: fast | fast_grasp | fast_nsga2. "
            f"Current optimizer '{method_opt}' is not allowed for Stage2."
        )

    set_global_seed(cfg.get("seed"))
    stamp = args.stamp or now_stamp()
    out_dir = os.path.join(args.run_dir, stamp)
    ensure_dir(out_dir)

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    t0 = time.perf_counter()
    docs = list(read_jsonl(args.input))
    rows = []
    for doc in tqdm(docs, desc="Summarizing", total=len(docs)):
        rows.append(summarize_one(doc, cfg))
    write_jsonl(preds_path, rows)
    t1 = time.perf_counter()

    # dump the config used
    import json
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # write selection time
    try:
        with open(os.path.join(out_dir, "time_select_seconds.txt"), "w", encoding="utf-8") as f:
            f.write(f"{t1 - t0:.6f}")
    except Exception:
        pass
    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
