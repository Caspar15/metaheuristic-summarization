from typing import Dict, List, Optional, Tuple


def _minmax_normalize(values: List[float]) -> List[float]:
    """Min-max normalize to [0, 1]."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo > 1e-9:
        return [(v - lo) / (hi - lo) for v in values]
    return [0.5] * len(values)


# ---------- original API (kept for backward compat) ----------

def combine_scores(
    features: Dict[str, List[float]],
    weights: Dict[str, float],
) -> List[float]:
    keys = list(next(iter(features.values())) if features else [])
    n = len(next(iter(features.values()))) if features else 0
    out = [0.0] * n
    for name, vals in features.items():
        w = float(weights.get(name, 0.0))
        if w == 0:
            continue
        for i, v in enumerate(vals):
            out[i] += w * v
    # normalize to [0,1]
    if out:
        mx = max(out) or 1.0
        out = [x / mx for x in out]
    return out


# ---------- improved version ----------

def combine_scores_v2(
    features: Dict[str, List[float]],
    weights: Dict[str, float],
    interactions: Optional[List[Tuple[str, str, float]]] = None,
    normalize: str = "minmax",
) -> List[float]:
    """Improved feature fusion with min-max normalization and optional
    multiplicative interaction terms.

    Parameters
    ----------
    features : dict
        ``{feature_name: [score_per_sentence]}``
    weights : dict
        ``{feature_name: weight}``
    interactions : list of (feat_a, feat_b, weight), optional
        Multiplicative interaction terms: ``w * f_a[i] * f_b[i]``.
    normalize : str
        ``"minmax"`` (default) or ``"max"`` (original behaviour).
    """
    n = len(next(iter(features.values()))) if features else 0
    if n == 0:
        return []

    # per-feature min-max normalization
    normed: Dict[str, List[float]] = {}
    for name, vals in features.items():
        w = float(weights.get(name, 0.0))
        if w == 0:
            normed[name] = [0.0] * n
            continue
        if normalize == "minmax":
            normed[name] = _minmax_normalize(vals)
        else:
            normed[name] = vals

    # linear combination
    out = [0.0] * n
    for name, vals in normed.items():
        w = float(weights.get(name, 0.0))
        if w == 0:
            continue
        for i in range(n):
            out[i] += w * vals[i]

    # interaction terms
    if interactions:
        for feat_a, feat_b, w_int in interactions:
            if feat_a in normed and feat_b in normed:
                for i in range(n):
                    out[i] += w_int * normed[feat_a][i] * normed[feat_b][i]

    # final normalization
    return _minmax_normalize(out)
