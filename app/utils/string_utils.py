"""String utility functions."""

from app.config import settings


def truncate_output(
    text: str,
    max_chars: int = settings.MAX_OUTPUT_CHARS,
    split_ratio: float = settings.OUTPUT_SPLIT_RATIO,
) -> str:
    """
    Truncate text to max_chars while preserving head and tail portions.

    If the text exceeds max_chars, it will be split into head and tail sections
    based on split_ratio, with a truncation marker in the middle.

    Args:
        text: The text to potentially truncate
        max_chars: Maximum character limit (default: from settings.MAX_OUTPUT_CHARS)
        split_ratio: Ratio of head vs tail (default: from settings.OUTPUT_SPLIT_RATIO)

    Returns:
        The original text if under max_chars, otherwise truncated text with
        head and tail sections separated by a truncation marker.

    Example:
        >>> text = "x" * 10000
        >>> result = truncate_output(text, max_chars=1000, split_ratio=0.6)
        >>> # Returns first 600 chars + marker + last 400 chars
    """

    if not text or len(text) <= max_chars:
        return text

    marker = f"\n[--- OUTPUT TRUNCATED | middle omitted | original length={len(text)} chars ---]\n"

    head_size = int(max_chars * split_ratio)
    tail_size = max_chars - head_size

    head = text[:head_size]
    tail = text[-tail_size:] if tail_size > 0 else ""

    return head + marker + tail
