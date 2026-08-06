"""Optimizer dispatch using dictionary mapping instead of if-elif chains."""

from typing import Dict, List, Optional

import numpy as np

from src.models.extractive.greedy import greedy_select
from src.models.extractive.grasp import grasp_select
from src.models.extractive.mmr import mmr_select
from src.objectives.evaluator import objective_from_spec

try:
    from src.models.extractive.encoder_rank import encoder_select  # type: ignore
except ImportError:
    def encoder_select(*args, **kwargs):  # type: ignore
        raise ImportError("Encoder ranking requires 'transformers' and 'torch'.")

try:
    from src.models.extractive.nsga2 import nsga2_select  # type: ignore
except ImportError:
    def nsga2_select(*args, **kwargs):  # type: ignore
        raise ImportError("nsga2 requires 'pymoo' to be installed.")

try:
    from src.models.extractive.fast_fused import (
        fast_fused_select,  # type: ignore
        fast_grasp_select,  # type: ignore
        fast_nsga2_select,  # type: ignore
    )
except ImportError:
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
    objective_spec: Optional[Dict] = None,
    min_words: int = 0,
    require_nonempty: bool = True,
    diagnostics: Optional[Dict] = None,
    coverage_matrix: Optional[np.ndarray] = None,
) -> List[int]:
    """Run the selected optimizer and return picked indices (relative to sub_sentences)."""

    method = method.lower()

    if not sub_sentences:
        return []

    if method in {"mmr", "nsga2"} and sub_sim is None:
        raise ValueError(
            f"{method} requires a similarity matrix; refusing to fall back "
            "to greedy because that would change the declared method."
        )

    shared_evaluator = None
    if method in {"greedy", "grasp", "mmr", "nsga2"}:
        shared_evaluator = objective_from_spec(
            sub_sentences,
            sub_scores,
            sub_sim,
            objective_spec,
            max_length=max_tokens,
            length_unit=unit,
            max_sentences=max_sents,
            min_words=min_words,
            require_nonempty=require_nonempty,
            coverage_matrix=coverage_matrix,
        )

    if method == "greedy":
        return greedy_select(
            sub_sentences, sub_scores, sub_sim, max_tokens,
            alpha=alpha, unit=unit, max_sentences=max_sents,
            evaluator=shared_evaluator,
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
            evaluator=shared_evaluator,
        )

    if method == "nsga2":
        obj = cfg.get("objectives", {})
        ocfg = cfg.get("optimizer", {})
        # AUDIT FIX (2026-07): pop_size / n_gen / seed were declared in the
        # YAML configs but NEVER read -- every NSGA-II run silently used the
        # function defaults (100/100).  They are now wired through, so the
        # values reported in the paper must come from the config actually used.
        # The blanket `except Exception -> greedy` fallback was also removed:
        # it could silently turn an NSGA-II experiment into a greedy one with
        # nothing recorded in metrics.csv.  Fail loudly instead.
        return nsga2_select(
            sub_sentences, sub_scores, sub_sim, max_tokens,
            lambda_importance=float(obj.get("lambda_importance", 1.0)),
            lambda_coverage=float(obj.get("lambda_coverage", 0.8)),
            lambda_redundancy=float(obj.get("lambda_redundancy", 0.7)),
            unit=unit,
            max_sentences=max_sents,
            coverage_method=str(obj.get("coverage_method", "max")),
            pop_size=int(ocfg.get("pop_size", 100)),
            n_gen=int(ocfg.get("n_gen", 100)),
            seed=cfg.get("seed"),
            importance_aggregation=str(
                (objective_spec or {}).get("importance_aggregation", "sum")
            ),
            min_words=min_words,
            require_nonempty=require_nonempty,
            evaluator=shared_evaluator,
            diagnostics=diagnostics,
        )

    if method == "mmr":
        ocfg = cfg.get("optimizer", {})
        return mmr_select(
            sub_sentences,
            sub_scores,
            sub_sim,
            max_tokens,
            lambda_relevance=float(ocfg.get("lambda_relevance", 0.7)),
            unit=unit,
            max_sentences=max_sents,
            evaluator=shared_evaluator,
            diagnostics=diagnostics,
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
            device=bert_cfg.get("device"),
            batch_size=int(bert_cfg.get("batch_size", 16)),
            max_model_tokens=int(bert_cfg.get("max_model_tokens", 256)),
            revision=bert_cfg.get("revision"),
        )

    if method in ("fast", "fast_fused", "tfidf_fused"):
        fcfg = cfg.get("fusion", {})
        w_base = float(fcfg.get("w_base", 0.5))
        w_sem = float(fcfg.get("w_bert", 0.5))
        alpha_f = float(cfg.get("redundancy", {}).get("lambda", 0.7))
        return fast_fused_select(
            sub_sentences, sub_scores, max_tokens,
            w_base=w_base, w_sem=w_sem, alpha=alpha_f,
            unit=unit, max_sentences=max_sents,
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
        # WARNING (2026-07 audit): the key is called `w_bert`, but in
        # fast_fused.py it weights a **TF-IDF centroid** score, not a PLM one.
        # No PLM is involved anywhere in the Stage-2 fusion path.  This is the
        # real reason the "remove BERT" ablation moves ROUGE-1 by only 0.0005.
        # See CODE_AUDIT_IEEE_Access.md finding F-3 before reporting these
        # weights as PLM fusion weights in the paper.
        w_sem = float(fcfg.get("w_plm", fcfg.get("w_bert", 0.5)))
        obj = cfg.get("objectives", {})
        ocfg = cfg.get("optimizer", {})
        return fast_nsga2_select(
            sub_sentences, sub_scores, max_tokens,
            w_base=w_base, w_sem=w_sem,
            unit=unit, max_sentences=max_sents,
            lambda_importance=float(obj.get("lambda_importance", 1.0)),
            lambda_coverage=float(obj.get("lambda_coverage", 0.8)),
            lambda_redundancy=float(obj.get("lambda_redundancy", 0.7)),
            pop_size=int(ocfg.get("pop_size", 100)),
            n_gen=int(ocfg.get("n_gen", 100)),
            seed=cfg.get("seed"),
        )

    raise ValueError(
        f"Unknown optimizer method {method!r}; refusing to silently run greedy."
    )
