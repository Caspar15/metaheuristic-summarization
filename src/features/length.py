from __future__ import annotations

from typing import List


def unique_length(tokens: List[str]) -> int:
    """Return count of unique tokens in the sentence."""
    return len(set(tokens))

