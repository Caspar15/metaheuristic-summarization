import argparse
import os
from typing import Dict, List

import numpy as np

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
from src.models.extractive.greedy import greedy_select
from src.models.extractive.grasp import grasp_select
from src.models.extractive.nsga2 import nsga2_select


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


def summarize_one(doc: Dict, cfg: Dict) -> Dict:
    sentences: List[str] = doc.get("sentences", [])
    highlights: str = doc.get("highlights", "")

    # base scores: feature-based (supervised scoring removed)
    base_scores = build_base_scores(sentences, cfg)

    rep_cfg = cfg.get("representations", {})
    sim = None
    if bool(rep_cfg.get("use", True)) and len(sentences) > 0:
        method = rep_cfg.get("method", "tfidf")
        vec = SentenceVectors(method=method)
        X = vec.fit_transform(sentences)
        sim = cosine_similarity_matrix(X)

    max_tokens = int(cfg.get("length_control", {}).get("max_tokens", 100))
    alpha = float(cfg.get("redundancy", {}).get("lambda", 0.7))

    # candidate pool: restrict selection to top-k if enabled
    k = int(cfg.get("candidates", {}).get("k", min(15, len(sentences))))
    use_cand = bool(cfg.get("candidates", {}).get("use", True))
    cand_idx = topk_by_score(base_scores, k)
    if use_cand and cand_idx:
        sub_sentences = [sentences[i] for i in cand_idx]
        sub_scores = [base_scores[i] for i in cand_idx]
        sub_sim = None
        if sim is not None:
            import numpy as _np
            sub_sim = sim[_np.ix_(cand_idx, cand_idx)]
    else:
        sub_sentences = sentences
        sub_scores = base_scores
        sub_sim = sim

    method_opt = cfg.get("optimizer", {}).get("method", "greedy").lower()
    if method_opt == "greedy":
        picked_sub = greedy_select(sub_sentences, sub_scores, sub_sim, max_tokens, alpha=alpha)
    elif method_opt == "grasp":
        picked_sub = grasp_select(
            sub_sentences, sub_scores, sub_sim, max_tokens, alpha=alpha, iters=10, seed=cfg.get("seed")
        )
    elif method_opt == "nsga2":
        if sub_sim is None:
            print("Warning: representations.use=false; NSGA-II requires similarity. Falling back to greedy.")
            picked_sub = greedy_select(sub_sentences, sub_scores, None, max_tokens, alpha=alpha)
        else:
            try:
                picked_sub = nsga2_select(
                    sub_sentences,
                    sub_scores,
                    sub_sim,
                    max_tokens,
                    lambda_importance=float(cfg.get("objectives", {}).get("lambda_importance", 1.0)),
                    lambda_coverage=float(cfg.get("objectives", {}).get("lambda_coverage", 0.8)),
                    lambda_redundancy=float(cfg.get("objectives", {}).get("lambda_redundancy", 0.7)),
                )
            except ImportError as e:
                print(f"Warning: pymoo not available for NSGA-II, falling back to greedy: {e}")
                picked_sub = greedy_select(sub_sentences, sub_scores, sub_sim, max_tokens, alpha=alpha)
            except Exception as e:
                print(f"Warning: NSGA-II optimization failed, falling back to greedy: {e}")
                picked_sub = greedy_select(sub_sentences, sub_scores, sub_sim, max_tokens, alpha=alpha)
    else:
        picked_sub = greedy_select(sub_sentences, sub_scores, sub_sim, max_tokens, alpha=alpha)

    # map back to original indices if we used a candidate subset
    if use_cand and cand_idx:
        selected = sorted(cand_idx[i] for i in picked_sub)
    else:
        selected = sorted(picked_sub)

    # keep original order
    selected.sort()
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
        cfg.setdefault("optimizer", {})
        cfg["optimizer"]["method"] = args.optimizer
    set_global_seed(cfg.get("seed"))
    stamp = args.stamp or now_stamp()
    out_dir = os.path.join(args.run_dir, stamp)
    ensure_dir(out_dir)

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    rows = []
    for doc in read_jsonl(args.input):
        rows.append(summarize_one(doc, cfg))
    write_jsonl(preds_path, rows)

    # also dump the config used
    import json
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()
