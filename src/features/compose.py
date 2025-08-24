from typing import Dict, List


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

