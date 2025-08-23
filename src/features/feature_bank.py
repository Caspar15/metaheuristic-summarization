from __future__ import annotations

from typing import Any, Dict, List, Tuple

import yaml

from src.features.tf import tf_score
from src.features.tf_isf import compute_isf, tf_isf_score
from src.features.length import unique_length
from src.features.position import position_scores


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def min_max_norm(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def build_features_for_doc(
    sentences: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Compute per-sentence features and total scores.

    Returns tuple of:
    - features list aligned with sentences: {features: {...}, score: float}
    - scores list
    """
    fcfg = cfg.get("features", {})
    weights = fcfg.get("weights", {"tf": 0.4, "tf_isf": 0.4, "length": 0.1, "position": 0.1})

    tokens_list: List[List[str]] = [s.get("tokens", []) for s in sentences]

    # TF
    tf_use = fcfg.get("tf", {}).get("use", True)
    tf_norm = fcfg.get("tf", {}).get("norm", "none")
    tf_vals = [tf_score(t, norm=tf_norm) for t in tokens_list] if tf_use else [0.0] * len(tokens_list)

    # TF-ISF
    tfisf_use = fcfg.get("tf_isf", {}).get("use", True)
    ngram = int(fcfg.get("tf_isf", {}).get("ngram", 1))  # ngram>1 not used in MVP
    smooth = float(fcfg.get("tf_isf", {}).get("smooth", 1.0))
    isf = compute_isf(tokens_list, smooth=smooth) if tfisf_use else {}
    tfisf_vals = [tf_isf_score(t, isf, norm=tf_norm) for t in tokens_list] if tfisf_use else [0.0] * len(tokens_list)

    # Length (unique tokens), min-max normalize
    len_use = fcfg.get("length", {}).get("use", True)
    lengths_raw = [float(unique_length(t)) for t in tokens_list]
    length_vals = min_max_norm(lengths_raw) if len_use else [0.0] * len(tokens_list)

    # Position
    pos_use = fcfg.get("position", {}).get("use", True)
    lead_bonus = float(fcfg.get("position", {}).get("lead_bonus", 0.2))
    position_vals = position_scores(len(tokens_list), lead_bonus=lead_bonus) if pos_use else [0.0] * len(tokens_list)

    feats: List[Dict[str, Any]] = []
    scores: List[float] = []
    for i in range(len(tokens_list)):
        f = {
            "tf": tf_vals[i],
            "tf_isf": tfisf_vals[i],
            "length": length_vals[i],
            "position": position_vals[i],
        }
        score = (
            weights.get("tf", 0.0) * f["tf"]
            + weights.get("tf_isf", 0.0) * f["tf_isf"]
            + weights.get("length", 0.0) * f["length"]
            + weights.get("position", 0.0) * f["position"]
        )
        feats.append({"features": f, "score": float(score)})
        scores.append(float(score))
    return feats, scores

