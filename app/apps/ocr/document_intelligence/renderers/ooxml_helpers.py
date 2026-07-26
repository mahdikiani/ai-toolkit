"""
Raw Open XML helpers for Word features python-docx has no API for.

python-docx wraps a large chunk of the OOXML spec, but field codes (PAGE,
NUMPAGES, TOC, ...), real hyperlinks, and footnotes are not among them —
these all require building the underlying ``w:fldChar``/``w:instrText``/
``w:hyperlink``/etc. elements by hand. This module is the one place that
does that, so callers stay focused on document structure rather than XML
plumbing.
"""

from __future__ import annotations

from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def add_field_run(paragraph: object, field_code: str, cached_text: str = "") -> None:
    """
    Append a real Word field (``PAGE``, ``NUMPAGES``, ...) to ``paragraph``.

    using the standard OOXML begin/instrText/separate/cached-result/end
    run sequence. ``cached_text`` is what's displayed until Word recalculates
    the field (typically on open/print) — without it the field would render
    as blank in viewers that don't auto-update, so a best-effort value (e.g.
    the page number seen at render time) keeps the document sensible even
    if it's never refreshed.
    """
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")

    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {field_code} "

    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")

    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")

    paragraph._p.append(_run_wrapping(begin))
    paragraph._p.append(_run_wrapping(instr))
    paragraph._p.append(_run_wrapping(separate))
    if cached_text:
        cached = OxmlElement("w:t")
        cached.set(qn("xml:space"), "preserve")
        cached.text = cached_text
        paragraph._p.append(_run_wrapping(cached))
    paragraph._p.append(_run_wrapping(end))


def _run_wrapping(child: object) -> OxmlElement:
    run = OxmlElement("w:r")
    run.append(child)
    return run
