"""Split long content into ordered, paragraph-aligned chunks."""

from __future__ import annotations


def split_into_chunks(content: str, max_chars: int) -> list[str]:
    """
    Split *content* into ordered chunks of at most *max_chars* characters.

    Splits along paragraph boundaries (blank lines) so a chunk never cuts
    a paragraph in half, except when a single paragraph itself exceeds
    max_chars, in which case that one paragraph is hard-split.
    """
    if not content:
        return []
    if len(content) <= max_chars:
        return [content]

    paragraphs = content.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.extend(
                para[i : i + max_chars] for i in range(0, len(para), max_chars)
            )
            continue

        sep_len = 2 if current else 0
        if current and current_len + sep_len + para_len > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
            sep_len = 0

        current.append(para)
        current_len += sep_len + para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks
