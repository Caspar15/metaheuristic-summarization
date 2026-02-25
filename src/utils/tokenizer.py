def count_tokens(text: str) -> int:
    """Count whitespace-delimited tokens in *text*."""
    return len((text or "").split())
