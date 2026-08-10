from collections import defaultdict
from typing import List, Mapping, Sequence
import math


# ---------- original API (kept for backward compat) ----------

def position_scores(sentences: List[str]) -> List[float]:
    n = len(sentences)
    if n == 0:
        return []
    # Higher score for earlier sentences (descending linear)
    return [1.0 - (i / max(1, n - 1)) if n > 1 else 1.0 for i in range(n)]


# ---------- improved version ----------

def position_scores_v2(
    sentences: List[str],
    method: str = "inverse",
    decay: float = 0.1,
) -> List[float]:
    """Position scoring with configurable decay functions.

    Parameters
    ----------
    method : str
        ``"linear"``  – original linear decay ``1 - i/(n-1)``
        ``"inverse"`` – ``1 / (1 + i)`` (stronger lead bias)
        ``"exponential"`` – ``exp(-decay * i)`` (configurable)
    decay : float
        Decay rate for the exponential method.
    """
    normalized_method = (method or "").strip().lower()
    if normalized_method not in {"linear", "inverse", "exponential"}:
        raise ValueError(f"unknown position scoring method: {method!r}")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("position exponential decay must be finite and non-negative")

    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    raw: List[float] = []
    for i in range(n):
        if normalized_method == "linear":
            raw.append(1.0 - (i / (n - 1)))
        elif normalized_method == "inverse":
            raw.append(1.0 / (1.0 + i))
        elif normalized_method == "exponential":
            raw.append(math.exp(-decay * i))

    # normalize to [0, 1]
    mx = max(raw)
    return [s / mx for s in raw] if mx > 0 else raw


def document_position_scores(
    sentence_records: Sequence[Mapping],
    *,
    version: str = "v1",
    method: str = "inverse",
    decay: float = 0.1,
) -> List[float]:
    """Score within-document positions without flattening document boundaries.

    Canonical records must supply a non-empty ``document_id`` and consecutive
    zero-based ``document_position`` values for every document. Refusing
    legacy/unavailable provenance is intentional: silently treating a flat
    row as one document would recreate F-25 under a more reassuring label.
    """

    if not sentence_records:
        return []
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for flat_index, record in enumerate(sentence_records):
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                "document-scoped position requires canonical document_id provenance"
            )
        position = record.get("document_position")
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError(
                "document-scoped position requires non-negative integer "
                "document_position values"
            )
        groups[document_id].append((flat_index, position))

    scores = [0.0] * len(sentence_records)
    normalized_version = (version or "v1").strip().lower()
    for document_id, members in groups.items():
        observed = sorted(position for _, position in members)
        expected = list(range(len(members)))
        if observed != expected:
            raise ValueError(
                f"document {document_id!r} positions must be consecutive "
                f"zero-based values; observed {observed[:10]!r}"
            )
        ordered = sorted(members, key=lambda item: item[1])
        placeholders = [""] * len(ordered)
        if normalized_version == "v2":
            document_scores = position_scores_v2(
                placeholders, method=method, decay=decay
            )
        elif normalized_version == "v1":
            document_scores = position_scores(placeholders)
        else:
            raise ValueError(f"unknown position feature version: {version!r}")
        for (flat_index, _), score in zip(ordered, document_scores):
            scores[flat_index] = score
    return scores
