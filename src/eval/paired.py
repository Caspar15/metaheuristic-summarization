"""Paired uncertainty estimates for per-document evaluation scores."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np


def paired_bootstrap_difference(
    system_scores: Sequence[float],
    reference_scores: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 2024,
    confidence: float = 0.95,
) -> Dict[str, float | int]:
    """Estimate a paired mean difference and percentile confidence interval.

    Rows are resampled as pairs.  The returned difference is always
    ``system - reference``.  A finite-sample correction keeps the two-sided
    bootstrap p-value away from exactly zero.
    """

    system = np.asarray(system_scores, dtype=float)
    reference = np.asarray(reference_scores, dtype=float)
    if system.ndim != 1 or reference.ndim != 1 or system.shape != reference.shape:
        raise ValueError("paired score arrays must be one-dimensional and aligned")
    if system.size == 0:
        raise ValueError("paired bootstrap requires at least one row")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not np.all(np.isfinite(system)) or not np.all(np.isfinite(reference)):
        raise ValueError("paired scores contain non-finite values")

    differences = system - reference
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples, dtype=float)
    # Batch the index matrix so full-validation runs do not allocate an
    # n_resamples x n_documents array at once.
    batch_size = min(512, n_resamples)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, differences.size, size=(stop - start, differences.size))
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    nonpositive = int(np.count_nonzero(bootstrap_means <= 0.0))
    nonnegative = int(np.count_nonzero(bootstrap_means >= 0.0))
    p_value = min(
        1.0,
        2.0
        * min(nonpositive + 1, nonnegative + 1)
        / (n_resamples + 1),
    )
    return {
        "n_pairs": int(differences.size),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "mean_difference": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value_two_sided": float(p_value),
    }


def holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    """Return Holm step-down adjusted p-values keyed like the input."""

    if not p_values:
        return {}
    checked = {}
    for key, value in p_values.items():
        number = float(value)
        if not np.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"invalid p-value for {key!r}: {value!r}")
        checked[str(key)] = number

    ordered = sorted(checked.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, (total - rank) * value)
        adjusted[key] = min(1.0, running)
    return adjusted
