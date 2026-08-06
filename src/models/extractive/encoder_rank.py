"""Sentence-encoder ranking used by both candidate routing and selection."""

import os
from typing import Any, Dict, List, Optional, Tuple

from src.utils.tokenizer import count_tokens


def _ensure_imports() -> None:
    try:
        import torch  # noqa: F401
        from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Encoder ranking requires the optional 'torch' and 'transformers' packages."
        ) from exc


_MODEL_CACHE: dict = {}


def load_encoder(
    model_name: str,
    device: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
):
    """Load and cache one tokenizer/model pair for the full run.

    Publication timing must report one-off loading separately from warm
    per-document inference. Use :func:`clear_encoder_cache` between cold-load
    benchmark configurations.
    """

    _ensure_imports()
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (model_name, revision, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    use_fast = "xlnet" not in model_name.lower()
    common_kwargs: Dict[str, Any] = {"token": token}
    if revision is not None:
        common_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=use_fast, **common_kwargs
    )

    if "roberta" in model_name.lower():
        try:
            config = AutoConfig.from_pretrained(model_name, **common_kwargs)
            config.add_pooling_layer = False  # type: ignore[attr-defined]
            model = AutoModel.from_pretrained(
                model_name, config=config, **common_kwargs
            )
        except (AttributeError, TypeError, ValueError):
            model = AutoModel.from_pretrained(model_name, **common_kwargs)
    else:
        model = AutoModel.from_pretrained(model_name, **common_kwargs)

    model.eval()
    model.to(device)
    _MODEL_CACHE[key] = (tokenizer, model, device)
    return _MODEL_CACHE[key]


def clear_encoder_cache() -> None:
    """Drop cached encoders (for cold-load benchmarks and isolated tests)."""

    _MODEL_CACHE.clear()


def _effective_max_tokens(tokenizer, requested: int) -> int:
    if requested < 2:
        raise ValueError("max_model_tokens must be at least 2")
    model_limit = getattr(tokenizer, "model_max_length", requested)
    if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 1_000_000:
        model_limit = requested
    return min(requested, model_limit)


def _sentence_embeddings(
    sentences: List[str],
    model_name: str,
    device: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    batch_size: int = 16,
    max_model_tokens: int = 256,
) -> Tuple[Any, Dict[str, Any]]:
    """Batch-encode sentences using the checkpoint's SBERT pooling contract.

    ``all-MiniLM-L6-v2`` is a SentenceTransformer checkpoint whose module
    graph is Transformer -> attention-mask-aware mean Pooling -> Normalize.
    Loading its Transformer backbone directly is equivalent only when both
    post-processing steps are reproduced here.  Returning unit vectors also
    lets downstream code use a dot product as an auditable cosine matrix.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    import torch

    tokenizer, model, resolved_device = load_encoder(
        model_name, device=device, token=token, revision=revision
    )
    effective_max = _effective_max_tokens(tokenizer, max_model_tokens)

    embeddings: List[torch.Tensor] = []
    input_tokens = 0
    truncated_sentences = 0
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        untruncated = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        token_rows = untruncated["input_ids"]
        lengths = [len(row) for row in token_rows]
        input_tokens += sum(lengths)
        truncated_sentences += sum(length > effective_max for length in lengths)

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=effective_max,
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded)
            last_hidden = output.last_hidden_state
            attention = encoded["attention_mask"].unsqueeze(-1)
            summed = (last_hidden * attention).sum(dim=1)
            denominator = attention.sum(dim=1).clamp(min=1)
            pooled = summed / denominator
            pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
            embeddings.append(pooled.detach().cpu())

    if not embeddings:
        return torch.empty((0, 0)), {
            "model_name": model_name,
            "model_revision": revision or model_name,
            "max_model_tokens": max_model_tokens,
            "effective_max_model_tokens": effective_max,
            "batch_size": batch_size,
            "estimated_cost": {
                "encoded_sentences": 0,
                "input_tokens_before_truncation": 0,
            },
            "truncated_sentences": 0,
            "truncation_rate": 0.0,
            "device": str(resolved_device),
            "pooling": "attention_mask_mean",
            "normalize_embeddings": True,
            "similarity": "normalized_dot_product_cosine",
        }

    config = getattr(model, "config", None)
    resolved_revision = (
        getattr(config, "_commit_hash", None)
        or revision
        or getattr(config, "name_or_path", None)
        or model_name
    )
    metadata = {
        "model_name": model_name,
        "model_revision": str(resolved_revision),
        "max_model_tokens": max_model_tokens,
        "effective_max_model_tokens": effective_max,
        "batch_size": batch_size,
        "estimated_cost": {
            "encoded_sentences": len(sentences),
            "input_tokens_before_truncation": input_tokens,
        },
        "truncated_sentences": truncated_sentences,
        "truncation_rate": truncated_sentences / len(sentences),
        "device": str(resolved_device),
        "pooling": "attention_mask_mean",
        "normalize_embeddings": True,
        "similarity": "normalized_dot_product_cosine",
    }
    return torch.cat(embeddings, dim=0), metadata


def _cosine_scores_to_centroid(embeddings) -> List[float]:
    if embeddings.size(0) == 0:
        return []
    centroid = embeddings.mean(dim=0, keepdim=True)
    normalized = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-12)
    normalized_centroid = centroid / (centroid.norm(dim=1, keepdim=True) + 1e-12)
    return (normalized * normalized_centroid).sum(dim=1).tolist()


def encoder_document_embeddings(
    sentences: List[str],
    *,
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 16,
    max_model_tokens: int = 256,
    revision: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Return normalized sentence embeddings and full model provenance."""

    _ensure_imports()
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    return _sentence_embeddings(
        sentences,
        model_name=model_name,
        device=device,
        token=token,
        revision=revision,
        batch_size=batch_size,
        max_model_tokens=max_model_tokens,
    )


def centroid_scores_from_embeddings(embeddings) -> List[float]:
    """Score normalized sentence embeddings against their normalized centroid."""

    return _cosine_scores_to_centroid(embeddings)


def cosine_matrix_from_embeddings(embeddings) -> "Any":
    """Return a deterministic cosine matrix for L2-normalized embeddings."""

    import numpy as np

    values = (
        embeddings.detach().cpu().numpy()
        if hasattr(embeddings, "detach")
        else np.asarray(embeddings)
    )
    if values.ndim != 2:
        raise ValueError("sentence embeddings must be a two-dimensional matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("sentence embeddings contain non-finite values")
    return np.asarray(values @ values.T, dtype=float)


def encoder_route_scores(
    sentences: List[str],
    *,
    model_name: str,
    device: Optional[str] = None,
    batch_size: int = 16,
    max_model_tokens: int = 256,
    revision: Optional[str] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """Score every input sentence before any candidate top-K is applied."""

    if not sentences:
        return [], {
            "model_name": model_name,
            "model_revision": revision or model_name,
            "max_model_tokens": max_model_tokens,
            "effective_max_model_tokens": max_model_tokens,
            "batch_size": batch_size,
            "estimated_cost": {
                "encoded_sentences": 0,
                "input_tokens_before_truncation": 0,
            },
            "truncated_sentences": 0,
            "truncation_rate": 0.0,
            "pooling": "attention_mask_mean",
            "normalize_embeddings": True,
            "similarity": "normalized_dot_product_cosine",
        }

    _ensure_imports()
    embeddings, metadata = encoder_document_embeddings(
        sentences,
        model_name=model_name,
        device=device,
        revision=revision,
        batch_size=batch_size,
        max_model_tokens=max_model_tokens,
    )
    return _cosine_scores_to_centroid(embeddings), metadata


def encoder_select(
    sentences: List[str],
    max_tokens: int,
    unit: str = "sentences",
    max_sentences: Optional[int] = 3,
    model_name: str = "bert-base-uncased",
    device: Optional[str] = None,
    batch_size: int = 16,
    max_model_tokens: int = 256,
    revision: Optional[str] = None,
) -> List[int]:
    """Rank sentences by encoder-centroid similarity under an output budget."""

    if not sentences:
        return []
    scores, _ = encoder_route_scores(
        sentences,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_model_tokens=max_model_tokens,
        revision=revision,
    )
    order = sorted(range(len(sentences)), key=lambda index: (-scores[index], index))

    if (unit or "sentences").lower() == "sentences":
        limit = (
            max_sentences
            if max_sentences is not None and max_sentences > 0
            else len(sentences)
        )
        picked = order[: int(limit)]
    else:
        budget = int(max_tokens)
        picked = []
        total = 0
        for index in order:
            tokens = count_tokens(sentences[index])
            if total + tokens <= budget:
                picked.append(index)
                total += tokens
    return sorted(picked)


# Backward-compatible alias.
bert_select = encoder_select
