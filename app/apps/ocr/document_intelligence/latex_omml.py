"""
LaTeX -> OMML (Office Math Markup Language) conversion for real Word equations.

latex2mathml only converts LaTeX to MathML. This module walks the MathML element
tree it produces and re-emits the equivalent OMML XML fragment, so formulas land
in the .docx as genuine, editable Word equation objects instead of styled text.
"""

from __future__ import annotations

import re
from xml.etree.ElementTree import Element  # ruff: ignore[suspicious-xml-etree-import]

from latex2mathml.converter import convert_to_element

_ENTITY_RE = re.compile(r"&#x([0-9A-Fa-f]+);")
_UPRIGHT_TAGS = {"mn", "mo", "mtext"}


class LatexConversionError(ValueError):
    """Raised when a LaTeX string cannot be converted to OMML."""

    @classmethod
    def empty(cls) -> LatexConversionError:
        """Build an error for an empty LaTeX string."""
        return cls("empty latex")

    @classmethod
    def from_exc(cls, exc: BaseException) -> LatexConversionError:
        """Build an error from an underlying conversion exception."""
        return cls(str(exc))


def latex_to_omml(latex: str) -> str:
    """
    Convert a LaTeX string to an ``<m:oMath>`` XML fragment (no namespace decl).

    Raises:
        LatexConversionError: if the LaTeX cannot be parsed.

    """
    latex = (latex or "").strip().strip("$").strip()
    if not latex:
        raise LatexConversionError.empty()
    try:
        root = convert_to_element(latex)
    except Exception as exc:
        raise LatexConversionError.from_exc(exc) from exc
    body = _convert_children(root)
    return f"<m:oMath>{body}</m:oMath>"


_PASSTHROUGH_TAGS = {"math", "mrow", "mstyle", "mpadded", "mphantom"}


def _tag(element: Element) -> str:
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


def _convert_children(element: Element) -> str:
    return "".join(_convert(child) for child in element)


def _convert_mfrac(element: Element) -> str:
    num, den = list(element)[:2]
    return (
        "<m:f><m:fPr><m:ctrlPr/></m:fPr>"
        f"<m:num>{_convert(num)}</m:num>"
        f"<m:den>{_convert(den)}</m:den></m:f>"
    )


def _convert_msup(element: Element) -> str:
    base, sup = list(element)[:2]
    e, s = _convert(base), _convert(sup)
    return f"<m:sSup><m:e>{e}</m:e><m:sup>{s}</m:sup></m:sSup>"


def _convert_msub(element: Element) -> str:
    base, sub = list(element)[:2]
    e, s = _convert(base), _convert(sub)
    return f"<m:sSub><m:e>{e}</m:e><m:sub>{s}</m:sub></m:sSub>"


def _convert_msubsup(element: Element) -> str:
    base, sub, sup = list(element)[:3]
    return (
        f"<m:sSubSup><m:e>{_convert(base)}</m:e>"
        f"<m:sub>{_convert(sub)}</m:sub>"
        f"<m:sup>{_convert(sup)}</m:sup></m:sSubSup>"
    )


def _convert_msqrt(element: Element) -> str:
    return (
        '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg/>'
        f"<m:e>{_convert_children(element)}</m:e></m:rad>"
    )


def _convert_mroot(element: Element) -> str:
    base, index = list(element)[:2]
    d, e = _convert(index), _convert(base)
    return f"<m:rad><m:radPr/><m:deg>{d}</m:deg><m:e>{e}</m:e></m:rad>"


def _convert_munder(element: Element) -> str:
    base, under = list(element)[:2]
    e, lim = _convert(base), _convert(under)
    return f"<m:limLow><m:e>{e}</m:e><m:lim>{lim}</m:lim></m:limLow>"


def _convert_mover(element: Element) -> str:
    base, over = list(element)[:2]
    e, lim = _convert(base), _convert(over)
    return f"<m:limUpp><m:e>{e}</m:e><m:lim>{lim}</m:lim></m:limUpp>"


def _convert_munderover(element: Element) -> str:
    base, under, over = list(element)[:3]
    inner_e, inner_lim = _convert(base), _convert(over)
    inner = f"<m:limUpp><m:e>{inner_e}</m:e><m:lim>{inner_lim}</m:lim></m:limUpp>"
    lim = _convert(under)
    return f"<m:limLow><m:e>{inner}</m:e><m:lim>{lim}</m:lim></m:limLow>"


def _convert_mtable(element: Element) -> str:
    rows = "".join(f"<m:mr>{_convert(row)}</m:mr>" for row in element)
    return f"<m:m><m:mPr><m:ctrlPr/></m:mPr>{rows}</m:m>"


def _convert_mtr(element: Element) -> str:
    return "".join(f"<m:e>{_convert(cell)}</m:e>" for cell in element)


def _convert_mspace(_element: Element) -> str:
    return _run(" ", upright=True)


def _convert_unknown(element: Element) -> str:
    """Fall back to a tag's own text plus any children, best-effort."""
    text = _decode_text(element.text)
    return (_run(text, upright=False) if text else "") + _convert_children(element)


_TAG_HANDLERS = {
    "mfrac": _convert_mfrac,
    "msup": _convert_msup,
    "msub": _convert_msub,
    "msubsup": _convert_msubsup,
    "msqrt": _convert_msqrt,
    "mroot": _convert_mroot,
    "munder": _convert_munder,
    "mover": _convert_mover,
    "munderover": _convert_munderover,
    "mtable": _convert_mtable,
    "mtr": _convert_mtr,
    "mtd": _convert_children,
    "mspace": _convert_mspace,
}


def _convert(element: Element) -> str:
    tag = _tag(element)

    if tag in _PASSTHROUGH_TAGS:
        return _convert_children(element)

    if tag == "mi":
        return _run(_decode_text(element.text), upright=False)

    if tag in _UPRIGHT_TAGS:
        return _run(_decode_text(element.text), upright=True)

    handler = _TAG_HANDLERS.get(tag)
    if handler is not None:
        return handler(element)

    return _convert_unknown(element)
