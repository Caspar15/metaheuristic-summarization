import argparse
import os
import time
from typing import Dict, List, Set

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
try:
    from src.models.extractive.bert_rank import bert_select  # type: ignore
except Exception:
    def bert_select(*args, **kwargs):  # type: ignore
        raise ImportError("BERT ranking requires 'transformers' and 'torch'.")
try:
    from src.models.extractive.nsga2 import nsga2_select  # type: ignore
except Exception:
    def nsga2_select(*args, **kwargs):  # type: ignore
        raise ImportError("nsga2 requires 'pymoo' to be installed.")
try:
    from src.models.extractive.fused import fused_mmr_select  # type: ignore
except Exception:
    def fused_mmr_select(*args, **kwargs):  # type: ignore
        raise ImportError("fused optimizer requires 'transformers' and 'torch'.")

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

    
try:
    from src.models.extractive.three_stage_xlnet import ThreeStageXLNetSelector  # type: ignore
except Exception:
    def ThreeStageXLNetSelector(*args, **kwargs):  # type: ignore
        raise ImportError("three_stage_xlnet requires 'transformers', 'torch', 'sentencepiece' and 'pymoo'.")
try:
    from src.models.extractive.three_stage_roberta import ThreeStageRobertaSelector  # type: ignore
except Exception:
    def ThreeStageRobertaSelector(*args, **kwargs):  # type: ignore
        raise ImportError("three_stage_roberta requires 'transformers', 'torch' and 'pymoo'.")



def build_base_scores(sentences: List[str], cfg: Dict) -> List[float]:
    f_importance = sentence_tf_isf_scores(sentences)
    f_len = length_scores(sentences)
    f_pos = position_scores(sentences)
    # 外部化特徵權重（若未提供，使用既有預設）
    feat_cfg = cfg.get("features", {}) or {}
    weights_cfg = feat_cfg.get("weights", {}) or {}
    weights = {
        "importance": float(weights_cfg.get("importance", cfg.get("objectives", {}).get("lambda_importance", 1.0))),
        "length": float(weights_cfg.get("length", 0.3)),
        "position": float(weights_cfg.get("position", 0.3)),
    }
    feats = {"importance": f_importance, "length": f_len, "position": f_pos}
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
                # fallback ignore centrality if sklearn not available
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
    base_scores = build_base_scores(sentences, cfg)

    rep_cfg = cfg.get("representations", {})
    sim = None
    if bool(rep_cfg.get("use", True)) and len(sentences) > 0:
        method = rep_cfg.get("method", "tfidf")
        vec = SentenceVectors(method=method)
        X = vec.fit_transform(sentences)
        sim = cosine_similarity_matrix(X)

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

    cand_idx = _build_candidate_union(sentences, base_scores, k, sources)

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
                cand_idx = _build_candidate_union(sentences, base_scores, used_k, sources)

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
    elif method_opt == "bert":
        # direct ranking by BERT sentence embeddings vs document centroid
        bert_cfg = cfg.get("bert", {})
        model_name = bert_cfg.get("model_name", "bert-base-uncased")
        try:
            picked_sub = bert_select(
                sub_sentences,
                max_tokens,
                unit=unit,
                max_sentences=max_sents,
                model_name=model_name,
            )
        except Exception as e:
            raise RuntimeError(
                f"BERT 排序執行失敗：{e}. 請確認 transformers/torch 是否已安裝，或改用 greedy/grasp/nsga2。"
            ) from e
    elif method_opt == "fused":
        # Fusion of base feature score and BERT score, then MMR via greedy with similarity from BERT embeddings.
        bert_cfg = cfg.get("bert", {})
        model_name = bert_cfg.get("model_name", "bert-base-uncased")
        fcfg = cfg.get("fusion", {})
        w_base = float(fcfg.get("w_base", 0.5))
        w_bert = float(fcfg.get("w_bert", 0.5))
        alpha_f = float(cfg.get("redundancy", {}).get("lambda", 0.7))
        try:
            picked_sub = fused_mmr_select(
                sub_sentences,
                sub_scores,
                max_tokens,
                w_base=w_base,
                w_bert=w_bert,
                alpha=alpha_f,
                unit=unit,
                max_sentences=max_sents,
                model_name=model_name,
            )
        except Exception as e:
            raise RuntimeError(
                f"Fused+MMR 排序執行失敗：{e}. 請確認 transformers/torch 是否已安裝，或改用其他 optimizer。"
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
    set_global_seed(cfg.get("seed"))
    stamp = args.stamp or now_stamp()
    out_dir = os.path.join(args.run_dir, stamp)
    ensure_dir(out_dir)

    preds_path = os.path.join(out_dir, "predictions.jsonl")
    t0 = time.perf_counter()
    rows = []
    for doc in read_jsonl(args.input):
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
