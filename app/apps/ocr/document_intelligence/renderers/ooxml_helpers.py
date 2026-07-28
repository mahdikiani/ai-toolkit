"""
Raw Open XML helpers for Word features python-docx has no API for.

python-docx wraps a large chunk of the OOXML spec, but field codes (PAGE,
NUMPAGES, TOC, ...), real hyperlinks, and footnotes are not among them --
these all require building the underlying
``w:fldChar``/``w:instrText``/``w:hyperlink``/etc. elements by hand. This
module is the one place that does that, so callers stay focused on
document structure rather than XML plumbing.
"""

from __future__ import annotations

from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def add_field_run(paragraph: object, field_code: str, cached_text: str = "") -> None:
    """
    Append a real Word field (``PAGE``, ``NUMPAGES``, ...) to ``paragraph``.

    Uses the standard OOXML begin/instrText/separate/cached-result/end run
    sequence. ``cached_text`` is what's displayed until Word recalculates
    the field (typically on open/print) -- without it the field would
    render as blank in viewers that don't auto-update, so a best-effort
    value (e.g. the page number seen at render time) keeps the document
    sensible even if it's never refreshed.
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


_HYPERLINK_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"


def add_hyperlink_run(paragraph: object, text: str, url: str) -> None:
    """
    Append a real Word hyperlink to ``paragraph``.

    A genuine ``w:hyperlink`` field the reader can Ctrl+click, not just
    blue underlined text. Requires a real relationship in the part's
    .rels -- ``python-docx`` has no API for this, so the relationship is
    added directly via ``paragraph.part.relate_to``, exactly as
    python-docx's own image-embed code does internally for ``r:embed``
    relationships.

    Sets the fa-IR proofing language on the run directly, rather than
    relying on the caller's usual ``_ensure_rtl`` pass: ``Paragraph.runs``
    in python-docx only walks direct ``w:r`` children of ``w:p``, not
    runs nested inside a ``w:hyperlink``, so a hyperlink run would
    otherwise be silently skipped by that fix-up.
    """
    part = paragraph.part
    r_id = part.relate_to(url, _HYPERLINK_REL_TYPE, is_external=True)

    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")  # Word's default hyperlink blue
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rtl = OxmlElement("w:rtl")
    rtl.set(qn("w:val"), "1")
    lang = OxmlElement("w:lang")
    lang.set(qn("w:val"), "fa-IR")
    lang.set(qn("w:bidi"), "fa-IR")
    # CT_RPr child order: color precedes u, both precede rtl, which
    # precedes lang -- appending out of this order is invalid OOXML.
    rpr.append(color)
    rpr.append(underline)
    rpr.append(rtl)
    rpr.append(lang)
    run.append(rpr)

    text_el = OxmlElement("w:t")
    text_el.set(qn("xml:space"), "preserve")
    text_el.text = text
    run.append(text_el)

    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_toc_field(paragraph: object) -> None:
    r"""
    Append a real Word TOC field (``TOC \o "1-3" \h \z \u``) to ``paragraph``.

    Word populates it from the document's Heading styles the first time
    it's opened/updated (F9) or printed -- there is no way to pre-compute
    page numbers here since real pagination only happens inside
    Word/a renderer, not python-docx.
    """
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")

    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = ' TOC \\o "1-3" \\h \\z \\u '

    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")

    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")

    paragraph._p.append(_run_wrapping(begin))
    paragraph._p.append(_run_wrapping(instr))
    paragraph._p.append(_run_wrapping(separate))
    placeholder = OxmlElement("w:t")
    placeholder.set(qn("xml:space"), "preserve")
    placeholder.text = "Right-click, choose Update Field to build the contents."
    paragraph._p.append(_run_wrapping(placeholder))
    paragraph._p.append(_run_wrapping(end))


def _run_wrapping(child: object) -> OxmlElement:
    run = OxmlElement("w:r")
    run.append(child)
    return run
