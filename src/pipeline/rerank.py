import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.utils.io import load_yaml, read_jsonl, write_jsonl


@dataclass
class RerankConfig:
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 20  # number of candidate summaries per doc to rerank
    normalize: str = "minmax"  # minmax | zscore
    weights: Dict[str, float] = None  # e.g., {"ce": 0.7, "base": 0.3}


def _load_rerank_config(cfg: Dict) -> RerankConfig:
    r = cfg.get("rerank", {}) or {}
    return RerankConfig(
        enabled=bool(r.get("enabled", False)),
        model_name=str(r.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        top_n=int(r.get("top_n", 20)),
        normalize=str(r.get("normalize", "minmax")),
        weights=dict(r.get("weights", {})),
    )


def _normalize(values: List[float], method: str) -> List[float]:
    if not values:
        return values
    if method == "zscore":
        import math

        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
        std = math.sqrt(var) or 1.0
        return [(v - mean) / std for v in values]
    # default min-max
    mn, mx = min(values), max(values)
    denom = (mx - mn) if mx > mn else 1.0
    return [(v - mn) / denom for v in values]


def rerank_candidates(
    document: str,
    candidates: List[Dict],
    base_key: str = "base_score",
    ce_scores: Optional[List[float]] = None,
    normalize: str = "minmax",
    weights: Optional[Dict[str, float]] = None,
) -> List[int]:
    """Return indices of candidates sorted by fused score (desc).

    Each candidate is a dict, expected keys include:
      - "summary": str
      - optional base metrics, e.g., base_key, length, redundancy, coverage

    ce_scores may be provided externally (when model inference is done elsewhere).
    """
    n = len(candidates)
    if n == 0:
        return []
    weights = weights or {"ce": 1.0}
    # gather base
    base_vals = [float(c.get(base_key, 0.0)) for c in candidates]
    base_vals = _normalize(base_vals, normalize)
    # gather ce
    if ce_scores is None:
        ce_vals = [0.0] * n
    else:
        ce_vals = _normalize([float(x) for x in ce_scores], normalize)
    fused = []
    for i in range(n):
        val = 0.0
        val += weights.get("ce", 0.0) * ce_vals[i]
        val += weights.get("base", 0.0) * base_vals[i]
        fused.append(val)
    order = sorted(range(n), key=lambda i: fused[i], reverse=True)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config yaml")
    ap.add_argument("--input", required=True, help="JSONL with candidates per doc")
    ap.add_argument("--out", required=True, help="JSONL with reranked selection per doc")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rc = _load_rerank_config(cfg)
    if not rc.enabled:
        raise RuntimeError("rerank.enabled is false; enable it in config to use this module.")

    rows_out = []
    for row in read_jsonl(args.input):
        doc_text = row.get("article") or row.get("document") or ""
        candidates = row.get("candidates", [])
        # Placeholder: CE model scoring is not yet implemented; use zeros
        ce_scores = [0.0 for _ in candidates]
        order = rerank_candidates(
            doc_text,
            candidates,
            base_key="base_score",
            ce_scores=ce_scores,
            normalize=rc.normalize,
            weights=rc.weights or {"ce": 1.0, "base": 0.0},
        )
        best = candidates[order[0]] if candidates else {}
        rows_out.append({
            "id": row.get("id"),
            "summary": best.get("summary", ""),
            "selected_indices": best.get("indices", []),
            "reference": row.get("reference") or row.get("highlights") or "",
            "rerank_order": order,
        })
    write_jsonl(args.out, rows_out)


if __name__ == "__main__":
    main()

