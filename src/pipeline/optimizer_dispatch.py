"""Optimizer dispatch using dictionary mapping instead of if-elif chains."""

from typing import Dict, List, Optional

import numpy as np

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
        raise ImportError("fast_fused requires scikit-learn.")
    def fast_grasp_select(*args, **kwargs):  # type: ignore
        raise ImportError("fast_grasp requires scikit-learn.")
    def fast_nsga2_select(*args, **kwargs):  # type: ignore
        raise ImportError("fast_nsga2 requires scikit-learn and pymoo.")


def dispatch_optimizer(
    method: str,
    sub_sentences: List[str],
    sub_scores: List[float],
    sub_sim: Optional[np.ndarray],
    max_tokens: int,
    cfg: Dict,
    alpha: float,
    unit: str,
    max_sents: Optional[int],
) -> List[int]:
    """Run the selected optimizer and return picked indices (relative to sub_sentences)."""

    method = method.lower()

    if method == "greedy":
        return greedy_select(
            sub_sentences, sub_scores, sub_sim, max_tokens,
            alpha=alpha, unit=unit, max_sentences=max_sents,
        )

    if method == "grasp":
        return grasp_select(
            sub_sentences, sub_scores, sub_sim, max_tokens,
            alpha=alpha,
            iters=int(cfg.get("grasp", {}).get("iters", 10)),
            rcl_ratio=float(cfg.get("grasp", {}).get("rcl_ratio", 0.3)),
            seed=cfg.get("seed"),
            unit=unit,
            max_sentences=max_sents,
        )

    if method == "nsga2":
        if sub_sim is None:
            print("Warning: representations.use=false; NSGA-II requires similarity. Falling back to greedy.")
            return greedy_select(
                sub_sentences, sub_scores, None, max_tokens,
                alpha=alpha, unit=unit, max_sentences=max_sents,
            )
        try:
            obj = cfg.get("objectives", {})
            return nsga2_select(
                sub_sentences, sub_scores, sub_sim, max_tokens,
                lambda_importance=float(obj.get("lambda_importance", 1.0)),
                lambda_coverage=float(obj.get("lambda_coverage", 0.8)),
                lambda_redundancy=float(obj.get("lambda_redundancy", 0.7)),
                unit=unit,
                max_sentences=max_sents,
                coverage_method=str(obj.get("coverage_method", "max")),
            )
        except (ImportError, Exception) as e:
            print(f"Warning: NSGA-II failed ({e}), falling back to greedy.")
            return greedy_select(
                sub_sentences, sub_scores, sub_sim, max_tokens,
                alpha=alpha, unit=unit, max_sentences=max_sents,
            )

    if method in ("bert", "roberta", "xlnet"):
        bert_cfg = cfg.get("bert", {})
        model_name = bert_cfg.get("model_name") or (
            "roberta-base" if method == "roberta"
            else ("xlnet-base-cased" if method == "xlnet" else "bert-base-uncased")
        )
        return encoder_select(
            sub_sentences, max_tokens,
            unit=unit, max_sentences=max_sents, model_name=model_name,
        )

    if method in ("fast", "fast_fused", "tfidf_fused"):
        fcfg = cfg.get("fusion", {})
        w_base = float(fcfg.get("w_base", 0.5))
        w_sem = float(fcfg.get("w_bert", 0.5))
        alpha_f = float(cfg.get("redundancy", {}).get("lambda", 0.7))
        try:
            return fast_fused_select(
                sub_sentences, sub_scores, max_tokens,
                w_base=w_base, w_sem=w_sem, alpha=alpha_f,
                unit=unit, max_sentences=max_sents,
            )
        except Exception as e:
            print(f"Warning: fast_fused failed ({e}); falling back to greedy")
            return greedy_select(
                sub_sentences, sub_scores, None, max_tokens,
                alpha=alpha, unit=unit, max_sentences=max_sents,
            )

    if method == "fast_grasp":
        fcfg = cfg.get("fusion", {})
        w_base = float(fcfg.get("w_base", 0.5))
        w_sem = float(fcfg.get("w_bert", 0.5))
        alpha_f = float(cfg.get("redundancy", {}).get("lambda", 0.7))
        return fast_grasp_select(
            sub_sentences, sub_scores, max_tokens,
            w_base=w_base, w_sem=w_sem, alpha=alpha_f,
            unit=unit, max_sentences=max_sents,
            iters=int(cfg.get("grasp", {}).get("iters", 15)),
            rcl_ratio=float(cfg.get("grasp", {}).get("rcl_ratio", 0.3)),
            seed=cfg.get("seed"),
        )

    if method == "fast_nsga2":
        fcfg = cfg.get("fusion", {})
        w_base = float(fcfg.get("w_base", 0.5))
        w_sem = float(fcfg.get("w_bert", 0.5))
        obj = cfg.get("objectives", {})
        return fast_nsga2_select(
            sub_sentences, sub_scores, max_tokens,
            w_base=w_base, w_sem=w_sem,
            unit=unit, max_sentences=max_sents,
            lambda_importance=float(obj.get("lambda_importance", 1.0)),
            lambda_coverage=float(obj.get("lambda_coverage", 0.8)),
            lambda_redundancy=float(obj.get("lambda_redundancy", 0.7)),
        )

    # fallback
    return greedy_select(
        sub_sentences, sub_scores, sub_sim, max_tokens,
        alpha=alpha, unit=unit, max_sentences=max_sents,
    )
