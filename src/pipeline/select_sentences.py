import argparse
import os
import time
from typing import Dict, List, Set
from tqdm import tqdm

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
from src.features.graph import compute_textrank_scores
from src.representations.sent_vectors import SentenceVectors
from src.representations.similarity import cosine_similarity_matrix
from src.selection.candidate_pool import topk_by_score
from src.models.extractive.greedy import greedy_select
from src.models.extractive.grasp import grasp_select
try:
    from src.models.extractive.encoder_rank import encoder_select  # type: ignore
except Exception:
    def encoder_select(*args, **kwargs):  # type: ignore
        raise ImportError("Encoder ranking requires 'transformers' and 'torch'.")
try:
    from src.models.extractive.nsga2 import nsga2_select  # type: ignore
except Exception:
    def nsga2_select(*args, **kwargs):  # type: ignore
        raise ImportError("nsga2 requires 'pymoo' to be installed.")

try:
    from src.models.extractive.fast_fused import (
        fast_fused_select,  # type: ignore
        fast_grasp_select,  # type: ignore
        fast_nsga2_select,  # type: ignore
    )
except Exception:
    def fast_fused_select(*args, **kwargs):  # type: ignore
        raise ImportError("fast_fused requires scikit-learn to be installed.")
    def fast_grasp_select(*args, **kwargs):  # type: ignore
        raise ImportError("fast_grasp requires scikit-learn to be installed.")
    def fast_nsga2_select(*args, **kwargs):  # type: ignore
        raise ImportError("fast_nsga2 requires scikit-learn and pymoo to be installed.")


def build_base_scores(sentences: List[str], cfg: Dict, similarity_matrix=None) -> List[float]:
    f_importance = sentence_tf_isf_scores(sentences)
    f_len = length_scores(sentences)
    f_pos = position_scores(sentences)
    # External feature weights: fall back to defaults when not provided
    feat_cfg = cfg.get("features", {}) or {}
    weights_cfg = feat_cfg.get("weights", {}) or {}
    weights = {
        "importance": float(weights_cfg.get("importance", cfg.get("objectives", {}).get("lambda_importance", 1.0))),
        "length": float(weights_cfg.get("length", 0.3)),
        "position": float(weights_cfg.get("position", 0.3)),
        "graph": float(weights_cfg.get("graph", 0.0)),  # default 0 if not specified
    }
    
    f_graph = []
    if weights["graph"] > 1e-9 and similarity_matrix is not None:
        try:
            f_graph = compute_textrank_scores(similarity_matrix)
        except Exception as e:
            print(f"Warning: Graph score computation failed: {e}")
            f_graph = [0.0] * len(sentences)
    else:
        f_graph = [0.0] * len(sentences)

    feats = {
        "importance": f_importance,
        "length": f_len,
        "position": f_pos,
        "graph": f_graph
    }
    return combine_scores(feats, weights)


def _topk_by_position(sentences: List[str], k: int) -> List[int]:
    # earlier sentences get higher position score
    pos = position_scores(sentences)
    idx = sorted(range(len(pos)), key=lambda i: pos[i], reverse=True)
    return idx[:k]


def _topk_by_centrality_tfidf(sentences: List[str], k: int) -> List[int]:
    # lightweight centrality via TF-IDF similarity (independent from representations.use)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(sentences) == 0:
        return []
    X = TfidfVectorizer(lowercase=True).fit_transform(sentences)
    sim = cosine_similarity(X)
    cent = sim.mean(axis=1)
    idx = sorted(range(len(sentences)), key=lambda i: float(cent[i]), reverse=True)
    idx = sorted(range(len(sentences)), key=lambda i: float(cent[i]), reverse=True)
    return idx[:k]


def _topk_by_graph_score(sentences: List[str], k: int, sim_matrix=None) -> List[int]:
    # PageRank / TextRank centrality
    if not sentences:
        return []
    
    scores = []
    if sim_matrix is not None:
         scores = compute_textrank_scores(sim_matrix)
    else:
        # Fallback to TF-IDF similarity if matrix not provided
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        X = TfidfVectorizer(lowercase=True).fit_transform(sentences)
        sim = cosine_similarity(X)
        scores = compute_textrank_scores(sim)

    idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    return idx[:k]


def _build_candidate_union(
    sentences: List[str], base_scores: List[float], k: int, sources: List[str], sim_matrix=None, threshold: float = 0.0
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
                # fallback ignore centrality if sklearn not available
                pass
        elif name in ("graph", "textrank"):
            try:
                cand.update(_topk_by_graph_score(sentences, k, sim_matrix, threshold=threshold))
            except Exception:
                pass
    if not cand:
        cand.update(topk_by_score(base_scores, k))
    # keep original order stability by sorting by index
    return sorted(cand)


def _greedy_oracle_indices(sentences: List[str], reference: str, max_tokens: int) -> List[int]:
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


def summarize_one(doc: Dict, cfg: Dict) -> Dict:
    sentences: List[str] = doc.get("sentences", [])
    highlights: str = doc.get("highlights", "")

    # base scores: feature-based (supervised scoring removed)
    # 1. Compute similarity first if needed for graph features
    rep_cfg = cfg.get("representations", {})
    sim = None
    if bool(rep_cfg.get("use", True)) and len(sentences) > 0:
        method = rep_cfg.get("method", "tfidf")
        vec = SentenceVectors(method=method)
        X = vec.fit_transform(sentences)
        sim = cosine_similarity_matrix(X)

    base_scores = build_base_scores(sentences, cfg, similarity_matrix=sim)

    lc = cfg.get("length_control", {})
    unit = (lc.get("unit", "tokens") or "tokens").lower()
    max_tokens = int(lc.get("max_tokens", 100))
    max_sents_limit = lc.get("max_sentences", None)
    max_sents = int(max_sents_limit) if (max_sents_limit is not None) else None
    alpha = float(cfg.get("redundancy", {}).get("lambda", 0.7))

    # candidate pool: soft/hard modes with multi-source union and optional recall target
    cand_cfg = cfg.get("candidates", {})
    k = int(cand_cfg.get("k", min(15, len(sentences))))
    use_cand = bool(cand_cfg.get("use", True))
    mode = (cand_cfg.get("mode", "hard") or "hard").lower()
    sources = cand_cfg.get("sources", ["score"]) or ["score"]
    soft_boost = float(cand_cfg.get("soft_boost", 1.05))

    # get threshold for candidates too
    g_thresh = float(cfg.get("graph_params", {}).get("threshold", 0.0))
    cand_idx = _build_candidate_union(sentences, base_scores, k, sources, sim_matrix=sim, threshold=g_thresh)

    # optional dynamic k to reach recall target (if reference available)
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
                cand_idx = _build_candidate_union(sentences, base_scores, used_k, sources, sim_matrix=sim)

    # apply mode
    if use_cand and cand_idx:
        if mode == "hard":
            sub_sentences = [sentences[i] for i in cand_idx]
            sub_scores = [base_scores[i] for i in cand_idx]
            sub_sim = None
            if sim is not None:
                import numpy as _np
                sub_sim = sim[_np.ix_(cand_idx, cand_idx)]
        else:  # soft mode: do not restrict, only boost candidate scores
            sub_sentences = sentences
            sub_scores = base_scores[:]
            for i in cand_idx:
                sub_scores[i] = float(sub_scores[i]) * soft_boost
            sub_sim = sim
    else:
        sub_sentences = sentences
        sub_scores = base_scores
        sub_sim = sim

    # --- MODIFICATION START ---
    # 新增的邏輯：如果長度單位是 'words'，則使用專門的貪婪演算法
    if unit == "words":
        max_words = int(lc.get("max_words", 400))
        picked_sub = []
        current_words = 0
        
        # 根據 sub_scores 對句子索引進行排序 (分數越高越好)
        sorted_indices = np.argsort(sub_scores)[::-1]
        
        for idx in sorted_indices:
            sentence_text = sub_sentences[idx]
            word_count = len(sentence_text.split())
            
            # 如果加入這句話不會超過字數上限
            if current_words + word_count <= max_words:
                picked_sub.append(idx)
                current_words += word_count
    
    # 如果長度單位不是 'words'，則沿用舊的、基於優化器的選擇邏輯
    else:
        method_opt = cfg.get("optimizer", {}).get("method", "greedy").lower()
        if method_opt == "greedy":
            picked_sub = greedy_select(
                sub_sentences, sub_scores, sub_sim, max_tokens, alpha=alpha, unit=unit, max_sentences=max_sents
            )
        elif method_opt == "grasp":
            picked_sub = grasp_select(
                sub_sentences,
                sub_scores,
                sub_sim,
                max_tokens,
                alpha=alpha,
                iters=10,
                seed=cfg.get("seed"),
                unit=unit,
                max_sentences=max_sents,
            )
        elif method_opt == "nsga2":
            if sub_sim is None:
                print("Warning: representations.use=false; NSGA-II requires similarity. Falling back to greedy.")
                picked_sub = greedy_select(
                    sub_sentences, sub_scores, None, max_tokens, alpha=alpha, unit=unit, max_sentences=max_sents
                )
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
                        unit=unit,
                        max_sentences=max_sents,
                    )
                except ImportError as e:
                    print(f"Warning: pymoo not available for NSGA-II, falling back to greedy: {e}")
                    picked_sub = greedy_select(
                        sub_sentences, sub_scores, sub_sim, max_tokens, alpha=alpha, unit=unit, max_sentences=max_sents
                    )
                except Exception as e:
                    print(f"Warning: NSGA-II optimization failed, falling back to greedy: {e}")
                    picked_sub = greedy_select(
                        sub_sentences, sub_scores, sub_sim, max_tokens, alpha=alpha, unit=unit, max_sentences=max_sents
                    )
        elif method_opt in ("bert", "roberta", "xlnet"):
            # direct ranking by encoder sentence embeddings vs document centroid
            bert_cfg = cfg.get("bert", {})
            model_name = bert_cfg.get("model_name") or ("roberta-base" if method_opt == "roberta" else ("xlnet-base-cased" if method_opt == "xlnet" else "bert-base-uncased"))
            try:
                picked_sub = encoder_select(
                    sub_sentences,
                    max_tokens,
                    unit=unit,
                    max_sentences=max_sents,
                    model_name=model_name,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Encoder ranking failed: {e}. Please ensure transformers and torch are installed, or switch to greedy/grasp/nsga2"
                ) from e
        elif method_opt in ("fast", "fast_fused", "tfidf_fused"):
            # Fast fusion using TF-IDF centroid score + base, with TF-IDF cosine MMR
            fcfg = cfg.get("fusion", {})
            w_base = float(fcfg.get("w_base", 0.5))
            w_sem = float(fcfg.get("w_bert", 0.5))  # reuse weight field
            alpha_f = float(cfg.get("redundancy", {}).get("lambda", 0.7))
            try:
                picked_sub = fast_fused_select(
                    sub_sentences,
                    sub_scores,
                    max_tokens,
                    w_base=w_base,
                    w_sem=w_sem,
                    alpha=alpha_f,
                    unit=unit,
                    max_sentences=max_sents,
                )
            except Exception as e:
                print(f"Warning: fast_fused failed ({e}); falling back to greedy")
                picked_sub = greedy_select(
                    sub_sentences, sub_scores, None, max_tokens, alpha=alpha, unit=unit, max_sentences=max_sents
                )
        elif method_opt in ("fast_grasp",):
            fcfg = cfg.get("fusion", {})
            w_base = float(fcfg.get("w_base", 0.5))
            w_sem = float(fcfg.get("w_bert", 0.5))
            alpha_f = float(cfg.get("redundancy", {}).get("lambda", 0.7))
            picked_sub = fast_grasp_select(
                sub_sentences,
                sub_scores,
                max_tokens,
                w_base=w_base,
                w_sem=w_sem,
                alpha=alpha_f,
                unit=unit,
                max_sentences=max_sents,
                iters=int(cfg.get("grasp", {}).get("iters", 15)),
                rcl_ratio=float(cfg.get("grasp", {}).get("rcl_ratio", 0.3)),
                seed=cfg.get("seed"),
            )
        elif method_opt in ("fast_nsga2",):
            fcfg = cfg.get("fusion", {})
            w_base = float(fcfg.get("w_base", 0.5))
            w_sem = float(fcfg.get("w_bert", 0.5))
            obj = cfg.get("objectives", {})
            picked_sub = fast_nsga2_select(
                sub_sentences,
                sub_scores,
                max_tokens,
                w_base=w_base,
                w_sem=w_sem,
                unit=unit,
                max_sentences=max_sents,
                lambda_importance=float(obj.get("lambda_importance", 1.0)),
                lambda_coverage=float(obj.get("lambda_coverage", 0.8)),
                lambda_redundancy=float(obj.get("lambda_redundancy", 0.7)),
            )
        else:
            picked_sub = greedy_select(
                sub_sentences, sub_scores, sub_sim, max_tokens, alpha=alpha, unit=unit, max_sentences=max_sents
            )
    # --- MODIFICATION END ---
    
    # map back to original indices if we used a hard candidate subset
    if use_cand and cand_idx and mode == "hard":
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

    # also dump the config used
    import json
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # write selection time for later aggregation
    try:
        with open(os.path.join(out_dir, "time_select_seconds.txt"), "w", encoding="utf-8") as f:
            f.write(f"{t1 - t0:.6f}")
    except Exception:
        pass
    print(f"Wrote predictions to {preds_path}")


if __name__ == "__main__":
    main()