"""
Tokenize inline Markdown emphasis into plain text + style segments.

Splits bold/italic/code/links into one run per style span, for renderers
that need that instead of dumping raw ``**text**`` into a single run.

Only used by the flow-based renderer's Markdown-sourced path (LLM output
like meeting minutes/summaries is full of ``**bold**``); OCR-extracted text
rarely contains literal Markdown emphasis so the absolute-layout renderer
does not need this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INLINE_TOKEN_RE = re.compile(
    r"\*\*(?P<bold>.+?)\*\*"
    r"|__(?P<bold2>.+?)__"
    r"|`(?P<code>[^`\n]+?)`"
    r"|(?<!\*)\*(?P<italic>[^*\n]+?)\*(?!\*)"
    r"|(?<!_)_(?P<italic2>[^_\n]+?)_(?!_)"
    r"|\[(?P<link_text>[^\]\n]+)\]\((?P<link_url>https?://[^\s)]+)\)"
)


@dataclass
class InlineSegment:
    """One contiguous run of text sharing the same inline style."""

    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False


def _segment_for_match(m: re.Match[str]) -> InlineSegment:
    """Build the styled segment for one regex match, by matched group."""
    if m.group("bold") is not None:
        return InlineSegment(m.group("bold"), bold=True)
    if m.group("bold2") is not None:
        return InlineSegment(m.group("bold2"), bold=True)
    if m.group("code") is not None:
        return InlineSegment(m.group("code"), code=True)
    if m.group("italic") is not None:
        return InlineSegment(m.group("italic"), italic=True)
    if m.group("italic2") is not None:
        return InlineSegment(m.group("italic2"), italic=True)
    # Links: rendered as their visible text only (no separate hyperlink
    # object) — good enough for a converted-to-Word summary/minutes document.
    return InlineSegment(m.group("link_text") or "")


def parse_inline_segments(text: str) -> list[InlineSegment]:
    """Split ``text`` into styled segments in reading order."""
    text = text or ""
    segments: list[InlineSegment] = []
    pos = 0
    for m in _INLINE_TOKEN_RE.finditer(text):
        if m.start() > pos:
            segments.append(InlineSegment(text[pos : m.start()]))
        segments.append(_segment_for_match(m))
        pos = m.end()
    if pos < len(text):
        segments.append(InlineSegment(text[pos:]))
    if not segments:
        segments.append(InlineSegment(""))
    return segments
