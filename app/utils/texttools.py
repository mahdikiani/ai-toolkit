"""Text processing utilities for normalization and manipulation."""


def normalize_text(text: str) -> str:
    """
    Normalize line endings and strip whitespace from text.

    Args:
        text: Input text string to normalize.

    Returns:
        Text with CRLF converted to LF and leading/trailing whitespace removed.
    """
    return text.replace("\r\n", "\n").strip()
