from typing import List, Optional, Tuple

import numpy as np


def _minmax_norm(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo = min(xs)
    hi = max(xs)
    if hi - lo < 1e-12:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _bert_scores_and_sim(
    sentences: List[str], model_name: str = "bert-base-uncased", device: Optional[str] = None
) -> Tuple[List[float], np.ndarray]:
    from src.models.extractive.bert_rank import _sentence_embeddings, _cosine_scores_to_centroid

    import os, hashlib, json, pathlib
    import torch

    # read token from environment if available
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

    # simple cache: hash(model_name + sentences) -> embs
    cache_root = os.environ.get("EMBED_CACHE_DIR", os.path.join("data", "cache", "embeddings"))
    pathlib.Path(cache_root).mkdir(parents=True, exist_ok=True)
    key_src = json.dumps({"model": model_name, "sents": sentences}, ensure_ascii=False)
    key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, f"{key}.npy")

    embs = None
    if os.path.exists(cache_path):
        try:
            import numpy as _np

            embs = torch.from_numpy(_np.load(cache_path))
        except Exception:
            embs = None

    if embs is None:
        with torch.inference_mode():
            embs = _sentence_embeddings(
                sentences, model_name=model_name, device=device, token=hf_token
            )
        # write cache
        try:
            import numpy as _np

            _np.save(cache_path, embs.detach().cpu().numpy())
        except Exception:
            pass
    # scores to centroid
    scores = _cosine_scores_to_centroid(embs)
    # pairwise cosine similarity
    a = embs / (embs.norm(dim=1, keepdim=True) + 1e-12)
    sim = (a @ a.T).detach().cpu().numpy()
    return scores, sim


def fused_mmr_select(
    sentences: List[str],
    base_scores: List[float],
    max_tokens: int,
    w_base: float = 0.5,
    w_bert: float = 0.5,
    alpha: float = 0.7,
    unit: str = "sentences",
    max_sentences: Optional[int] = 3,
    model_name: str = "bert-base-uncased",
) -> List[int]:
    from src.models.extractive.greedy import greedy_select

    if not sentences:
        return []
    bert_scores, sim = _bert_scores_and_sim(sentences, model_name=model_name)
    base_n = _minmax_norm(list(base_scores))
    bert_n = _minmax_norm(list(bert_scores))
    fused = [float(w_base) * base_n[i] + float(w_bert) * bert_n[i] for i in range(len(sentences))]
    picked = greedy_select(
        sentences,
        fused,
        sim,
        max_tokens,
        alpha=float(alpha),
        unit=unit,
        max_sentences=max_sentences,
    )
    return picked
