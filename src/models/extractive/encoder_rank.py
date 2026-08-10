"""Sentence-encoder ranking used by both candidate routing and selection."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
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

# This cache is an execution-only optimisation.  It is deliberately opt-in so
# an unrelated experiment cannot silently inherit artifacts from an earlier
# run.  Bump the contract whenever pooling, normalisation, tokenisation, or the
# serialized payload changes.
EMBEDDING_CACHE_ENV = "META_SUM_EMBEDDING_CACHE_DIR"
EMBEDDING_CACHE_CONTRACT_VERSION = "sbert_mean_pool_l2_npz_v1"


def _sentence_sequence_sha256(sentences: List[str]) -> str:
    """Hash an ordered sentence sequence without delimiter ambiguity."""

    digest = hashlib.sha256()
    digest.update(len(sentences).to_bytes(8, "big"))
    for sentence in sentences:
        payload = sentence.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _embedding_cache_identity(
    sentences: List[str],
    *,
    model_name: str,
    revision: Optional[str],
    device: Optional[str],
    batch_size: int,
    max_model_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    identity = {
        "cache_contract_version": EMBEDDING_CACHE_CONTRACT_VERSION,
        "sentence_count": len(sentences),
        "sentence_sequence_sha256": _sentence_sequence_sha256(sentences),
        "model_name": model_name,
        "revision": revision,
        "resolved_device": str(resolved_device),
        "batch_size": int(batch_size),
        "max_model_tokens": int(max_model_tokens),
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), identity


def _cache_path(cache_root: Path, key: str) -> Path:
    return cache_root / key[:2] / f"{key}.npz"


def _load_cached_embeddings(path: Path, *, key: str, identity: Dict[str, Any]):
    import numpy as np
    import torch

    try:
        with np.load(path, allow_pickle=False) as payload:
            embeddings = np.asarray(payload["embeddings"])
            metadata_bytes = np.asarray(payload["metadata_json"], dtype=np.uint8)
        stored = json.loads(metadata_bytes.tobytes().decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid embedding cache artifact {path}: {exc}") from exc
    if stored.get("cache_key") != key or stored.get("identity") != identity:
        raise ValueError(f"embedding cache identity mismatch: {path}")
    if embeddings.dtype != np.float32 or embeddings.ndim != 2:
        raise ValueError(f"embedding cache must contain a 2-D float32 matrix: {path}")
    if embeddings.shape[0] != identity["sentence_count"]:
        raise ValueError(f"embedding cache row count mismatch: {path}")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError(f"embedding cache contains non-finite values: {path}")
    metadata = stored.get("encoder_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"embedding cache has no encoder metadata: {path}")
    return torch.from_numpy(np.array(embeddings, copy=True)), metadata


def _write_cached_embeddings(
    path: Path,
    *,
    key: str,
    identity: Dict[str, Any],
    embeddings: Any,
    metadata: Dict[str, Any],
) -> None:
    import numpy as np

    values = np.ascontiguousarray(embeddings.detach().cpu().numpy(), dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != identity["sentence_count"]:
        raise ValueError("refusing to cache an invalid embedding matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("refusing to cache non-finite embeddings")
    stored = {
        "cache_key": key,
        "identity": identity,
        "encoder_metadata": metadata,
    }
    metadata_bytes = json.dumps(
        stored, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{key}.",
            suffix=".partial",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez(
                handle,
                embeddings=values,
                metadata_json=np.frombuffer(metadata_bytes, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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
            # This call is only used to count pre-truncation tokens.  The
            # Transformers warning about feeding >model_max_length tokens is
            # misleading here because these IDs are never passed to the
            # model; the second call below performs the actual truncation.
            verbose=False,
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
    cache_value = os.environ.get(EMBEDDING_CACHE_ENV)
    if not cache_value:
        return _sentence_embeddings(
            sentences,
            model_name=model_name,
            device=device,
            token=token,
            revision=revision,
            batch_size=batch_size,
            max_model_tokens=max_model_tokens,
        )

    cache_root = Path(cache_value).resolve()
    key, identity = _embedding_cache_identity(
        sentences,
        model_name=model_name,
        revision=revision,
        device=device,
        batch_size=batch_size,
        max_model_tokens=max_model_tokens,
    )
    path = _cache_path(cache_root, key)
    if path.exists():
        embeddings, metadata = _load_cached_embeddings(
            path, key=key, identity=identity
        )
        metadata = dict(metadata)
        metadata["embedding_cache"] = {
            "enabled": True,
            "status": "hit",
            "cache_key": key,
            "contract_version": EMBEDDING_CACHE_CONTRACT_VERSION,
        }
        return embeddings, metadata

    embeddings, metadata = _sentence_embeddings(
        sentences,
        model_name=model_name,
        device=device,
        token=token,
        revision=revision,
        batch_size=batch_size,
        max_model_tokens=max_model_tokens,
    )
    _write_cached_embeddings(
        path,
        key=key,
        identity=identity,
        embeddings=embeddings,
        metadata=metadata,
    )
    metadata = dict(metadata)
    metadata["embedding_cache"] = {
        "enabled": True,
        "status": "miss_written",
        "cache_key": key,
        "contract_version": EMBEDDING_CACHE_CONTRACT_VERSION,
    }
    return embeddings, metadata


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
