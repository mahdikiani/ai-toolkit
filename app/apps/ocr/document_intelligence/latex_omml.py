"""
LaTeX -> OMML (Office Math Markup Language) conversion for real Word equations.

latex2mathml only converts LaTeX to MathML. This module walks the MathML element
tree it produces and re-emits the equivalent OMML XML fragment, so formulas land
in the .docx as genuine, editable Word equation objects instead of styled text.
"""

from __future__ import annotations

import re

from latex2mathml.converter import convert_to_element

_ENTITY_RE = re.compile(r"&#x([0-9A-Fa-f]+);")
_UPRIGHT_TAGS = {"mn", "mo", "mtext"}


class LatexConversionError(ValueError):
    """Raised when a LaTeX string cannot be converted to OMML."""


def latex_to_omml(latex: str) -> str:
    """
    Convert a LaTeX string to an ``<m:oMath>`` XML fragment (no namespace decl).

    Raises:
        LatexConversionError: if the LaTeX cannot be parsed.
    """
    latex = (latex or "").strip().strip("$").strip()
    if not latex:
        raise LatexConversionError("empty latex")
    try:
        root = convert_to_element(latex)
    except Exception as exc:
        raise LatexConversionError(str(exc)) from exc
    body = _convert_children(root)
    return f"<m:oMath>{body}</m:oMath>"


def _tag(element) -> str:
    t = element.tag
    return t.split("}")[-1] if "}" in t else t


def _decode_text(text: str | None) -> str:
    if not text:
        return ""
    return _ENTITY_RE.sub(lambda m: chr(int(m.group(1), 16)), text)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _run(text: str, upright: bool = False) -> str:
    if not text:
        return ""
    rpr = "<m:rPr><m:nor/></m:rPr>" if upright else ""
    return f'<m:r>{rpr}<m:t xml:space="preserve">{_escape(text)}</m:t></m:r>'


def _convert_children(element) -> str:
    return "".join(_convert(child) for child in element)


def _convert(element) -> str:
    tag = _tag(element)

    if tag in ("math", "mrow", "mstyle", "mpadded", "mphantom"):
        return _convert_children(element)

    if tag == "mi":
        return _run(_decode_text(element.text), upright=False)

    if tag in _UPRIGHT_TAGS:
        return _run(_decode_text(element.text), upright=True)

    if tag == "mfrac":
        num, den = list(element)[:2]
        return (
            "<m:f><m:fPr><m:ctrlPr/></m:fPr>"
            f"<m:num>{_convert(num)}</m:num>"
            f"<m:den>{_convert(den)}</m:den></m:f>"
        )

    if tag == "msup":
        base, sup = list(element)[:2]
        return f"<m:sSup><m:e>{_convert(base)}</m:e><m:sup>{_convert(sup)}</m:sup></m:sSup>"

    if tag == "msub":
        base, sub = list(element)[:2]
        return f"<m:sSub><m:e>{_convert(base)}</m:e><m:sub>{_convert(sub)}</m:sub></m:sSub>"

    if tag == "msubsup":
        base, sub, sup = list(element)[:3]
        return (
            f"<m:sSubSup><m:e>{_convert(base)}</m:e>"
            f"<m:sub>{_convert(sub)}</m:sub>"
            f"<m:sup>{_convert(sup)}</m:sup></m:sSubSup>"
        )

    if tag == "msqrt":
        return (
            '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/>'
            f"<m:e>{_convert_children(element)}</m:e></m:rad>"
        )

    if tag == "mroot":
        base, index = list(element)[:2]
        return f"<m:rad><m:radPr/><m:deg>{_convert(index)}</m:deg><m:e>{_convert(base)}</m:e></m:rad>"

    if tag == "munder":
        base, under = list(element)[:2]
        return f"<m:limLow><m:e>{_convert(base)}</m:e><m:lim>{_convert(under)}</m:lim></m:limLow>"

    if tag == "mover":
        base, over = list(element)[:2]
        return f"<m:limUpp><m:e>{_convert(base)}</m:e><m:lim>{_convert(over)}</m:lim></m:limUpp>"

    if tag == "munderover":
        base, under, over = list(element)[:3]
        inner = f"<m:limUpp><m:e>{_convert(base)}</m:e><m:lim>{_convert(over)}</m:lim></m:limUpp>"
        return f"<m:limLow><m:e>{inner}</m:e><m:lim>{_convert(under)}</m:lim></m:limLow>"

    if tag == "mtable":
        rows = "".join(f"<m:mr>{_convert(row)}</m:mr>" for row in element)
        return f"<m:m><m:mPr><m:ctrlPr/></m:mPr>{rows}</m:m>"

    if tag == "mtr":
        return "".join(f"<m:e>{_convert(cell)}</m:e>" for cell in element)

    if tag == "mtd":
        return _convert_children(element)

    if tag == "mspace":
        return _run(" ", upright=True)

    # Unknown tag: fall back to its own text plus any children, best-effort.
    text = _decode_text(element.text)
    return (_run(text, upright=False) if text else "") + _convert_children(element)
