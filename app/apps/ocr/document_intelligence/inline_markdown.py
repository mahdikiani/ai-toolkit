"""
Tokenize inline Markdown emphasis into plain text + style segments.

Splits bold/italic/code/link/math spans into one segment per style span,
for renderers that need that instead of dumping raw ``**text**`` into a
single run. A link segment carries its URL (``InlineSegment.url``) so the
caller can emit a real ``w:hyperlink`` -- see renderers/ooxml_helpers.py's
``add_hyperlink_run`` -- instead of just the visible link text. A math
segment (``InlineSegment.math``) marks LaTeX meant mid-sentence (e.g.
``احتمال $p_i$ برابر است با``) so the caller can emit a real inline
``m:oMath`` -- see latex_omml.py -- instead of leaving the raw
``$p_i$`` text sitting inside an RTL paragraph, where it reliably produces
bidi word-reordering artifacts (mixed Persian/Latin/math tokens with no
directional isolation).

Used by the flow-based renderer (docx.py) for both its Markdown-sourced
path (LLM output like meeting minutes is full of ``**bold**``/links) and
OCR-extracted paragraph text, which occasionally contains a literal URL or
inline LaTeX the VLM transcribed as Markdown/dollar-delimited math.
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
    r"|\${1,2}(?P<math>[^$\n]+?)\${1,2}"
)


@dataclass
class InlineSegment:
    """One contiguous run of text sharing the same inline style."""

    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False
    url: str = ""  # non-empty -> renderers must emit a real w:hyperlink
    math: bool = False  # True -> .text is LaTeX for a real inline m:oMath


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
    if m.group("math") is not None:
        return InlineSegment(m.group("math"), math=True)
    return InlineSegment(m.group("link_text") or "", url=m.group("link_url") or "")


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
