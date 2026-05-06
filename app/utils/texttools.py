"""Text processing utilities."""


def normalize_text(text: str) -> str:
    """Normalize text by replacing CRLF with LF and stripping whitespace."""
    return text.replace("\r\n", "\n").strip()
