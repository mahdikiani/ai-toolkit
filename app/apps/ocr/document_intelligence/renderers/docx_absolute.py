"""Word Renderer — absolute layout, one floating text box per element.

The flow-based renderer (``docx.py``) lays elements out one after another
like normal typing, which reads fine but bears no visual resemblance to the
source page (no side-by-side boxes, no elements sitting where they actually
were). This renderer instead places every element in a VML floating text
box positioned at its real source-page bbox, so the output actually looks
like the original document.

VML (not the more modern DrawingML) is used deliberately: it has much wider
compatibility across Word and LibreOffice for simple absolutely-positioned
text boxes, and was verified end-to-end (RTL text, real tables, real OMML
formulas, real images, correct per-page position reset across page breaks)
by rendering prototypes through LibreOffice headless before writing this.

Trade-off accepted by the user: editing is less fluid (each box is a
separate floating shape, not one continuous flow), and decorative page
chrome (borders, background shapes) is not reconstructed — only the actual
detected content elements (text/table/formula/image) are positioned.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches

from ..ast import ASTNode, DocumentAST, PageAST
from ..latex_omml import LatexConversionError, latex_to_omml
from ..layout import LayoutType
from .docx import (
    FONT_MATH,
    _collect_header_footer,
    _resolve_fonts,
    _resolve_margin,
    _resolve_page_size,
    _set_section_footer,
    _set_section_header,
)

logger = logging.getLogger(__name__)

_VML_NS = 'xmlns:v="urn:schemas-microsoft-com:vml"'
_O_NS = 'xmlns:o="urn:schemas-microsoft-com:office:office"'
_R_NS = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
_M_NS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'

_SHAPE_TYPE_XML = (
    '<v:shapetype id="_x0000_t202_di" coordsize="21600,21600" o:spt="202" '
    'path="m,l,21600r21600,l21600,xe"/>'
)

# Minimum box size so near-zero-height/width bboxes (bad detections) still
# render something visible instead of a collapsed, invisible shape.
MIN_BOX_WIDTH_IN = 0.3
MIN_BOX_HEIGHT_IN = 0.15


def render_docx_absolute(ast: DocumentAST, pdf_data: bytes | None = None) -> BytesIO:
    """Render DocumentAST to .docx with every element absolutely positioned
    at its source-page bbox, reconstructing the original page layout."""
    font_cs, font_latin = _resolve_fonts(pdf_data)
    page_width_in, page_height_in = _resolve_page_size(ast)
    margin_in = _resolve_margin(page_width_in)

    doc = Document()
    _setup_minimal_style(doc, font_cs, font_latin)
    for section in doc.sections:
        section.page_width = Inches(page_width_in)
        section.page_height = Inches(page_height_in)
        section.top_margin = Inches(margin_in)
        section.bottom_margin = Inches(margin_in)
        section.left_margin = Inches(margin_in)
        section.right_margin = Inches(margin_in)

    if ast.title:
        doc.core_properties.title = ast.title

    header_text, footer_text = _collect_header_footer(ast)
    if header_text:
        _set_section_header(doc, header_text)
    if footer_text:
        _set_section_footer(doc, footer_text)

    body = doc.element.body
    shape_type_emitted = False
    prev_page = 0
    for page in ast.pages:
        if prev_page > 0 and page.page_number > prev_page:
            body.append(_page_break_xml())
        shape_type_emitted = _render_page_absolute(
            doc, body, page, font_cs, font_latin, shape_type_emitted
        )
        prev_page = page.page_number

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def _setup_minimal_style(doc: Document, font_cs: str, font_latin: str) -> None:
    """Set the document's base font; per-box formatting is applied inline
    since each box is independent XML, not a shared paragraph style."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    style = doc.styles["Normal"]
    rpr = style.element.get_or_add_rPr()
    rfonts = OxmlElement("w:rFonts")
    rfonts.set(qn("w:ascii"), font_latin)
    rfonts.set(qn("w:hAnsi"), font_latin)
    rfonts.set(qn("w:cs"), font_cs)
    rpr.append(rfonts)


def _page_break_xml():
    return parse_xml(
        f'<w:p {nsdecls("w")}><w:r><w:br w:type="page"/></w:r></w:p>'
    )


def _render_page_absolute(
    doc: Document,
    body,
    page: PageAST,
    font_cs: str,
    font_latin: str,
    shape_type_emitted: bool,
) -> bool:
    for node in page.nodes:
        if node.type in (LayoutType.header, LayoutType.footer):
            continue  # promoted to real Word header/footer, see _collect_header_footer
        xml = _render_node_absolute(
            doc, node, page, font_cs, font_latin, include_shapetype=not shape_type_emitted
        )
        if xml is not None:
            body.append(xml)
            shape_type_emitted = True
    return shape_type_emitted


def _bbox_to_inches(
    bbox: tuple[float, float, float, float], page: PageAST
) -> tuple[float, float, float, float]:
    dpi = page.page_dpi or 300.0
    x1, y1, x2, y2 = bbox
    left = x1 / dpi
    top = y1 / dpi
    width = max(MIN_BOX_WIDTH_IN, (x2 - x1) / dpi)
    height = max(MIN_BOX_HEIGHT_IN, (y2 - y1) / dpi)
    return left, top, width, height


def _render_node_absolute(
    doc: Document,
    node: ASTNode,
    page: PageAST,
    font_cs: str,
    font_latin: str,
    include_shapetype: bool,
) -> object | None:
    if node.type in (LayoutType.figure, LayoutType.chart):
        return _image_shape_xml(doc, node, page, include_shapetype)

    width_in, height_in = _bbox_to_inches(node.bbox, page)[2:]
    inner = _node_inner_xml(node, font_cs, font_latin, width_in, height_in)
    if inner is None:
        return None
    return _textbox_shape_xml(node.bbox, page, inner, include_shapetype)


def _node_inner_xml(
    node: ASTNode, font_cs: str, font_latin: str, width_in: float, height_in: float
) -> str | None:
    if node.type == LayoutType.title:
        return _paragraph_xml(
            node.text, bold=True, size_pt=16, font_cs=font_cs, font_latin=font_latin,
            width_in=width_in, height_in=height_in,
        )
    if node.type == LayoutType.heading:
        return _paragraph_xml(
            node.text, bold=True, size_pt=13, font_cs=font_cs, font_latin=font_latin,
            width_in=width_in, height_in=height_in,
        )
    if node.type in (LayoutType.paragraph, LayoutType.reference):
        return _paragraph_xml(
            node.text, font_cs=font_cs, font_latin=font_latin,
            width_in=width_in, height_in=height_in,
        )
    if node.type == LayoutType.list:
        items = node.children or [node]
        per_item_height = height_in / max(1, len(items))
        return "".join(
            _paragraph_xml(
                f"• {c.text}", font_cs=font_cs, font_latin=font_latin,
                width_in=width_in, height_in=per_item_height,
            )
            for c in items
        )
    if node.type == LayoutType.table:
        return _table_xml(node)
    if node.type == LayoutType.formula:
        return _formula_xml(node.latex, font_cs, font_latin)
    if node.type == LayoutType.code:
        return _paragraph_xml(
            node.text, font_latin="Courier New", font_cs="Courier New", size_pt=9,
            width_in=width_in, height_in=height_in,
        )
    return None


def _escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_CHAR_WIDTH_FACTOR = 0.50  # rough avg glyph width ≈ factor * size(pt) / 72in
_LINE_HEIGHT_FACTOR = 1.25  # rough line height ≈ factor * size(pt) / 72in
MIN_AUTOFIT_SIZE_PT = 6


def _autofit_size_pt(text: str, width_in: float, height_in: float, base_size_pt: int) -> int:
    """Shrink the font size so wrapped text is more likely to fit inside its
    box height, reducing (not fully eliminating — this is a rough
    proportional-font estimate, not real text layout) overlap with
    neighboring boxes positioned right below/beside it."""
    if not text or width_in <= 0 or height_in <= 0:
        return base_size_pt
    for size in range(base_size_pt, MIN_AUTOFIT_SIZE_PT - 1, -1):
        char_width_in = _CHAR_WIDTH_FACTOR * size / 72
        line_height_in = _LINE_HEIGHT_FACTOR * size / 72
        chars_per_line = max(1, int(width_in / char_width_in))
        lines_needed = max(1, -(-len(text) // chars_per_line))  # ceil div
        if lines_needed * line_height_in <= height_in:
            return size
    return MIN_AUTOFIT_SIZE_PT


def _paragraph_xml(
    text: str,
    *,
    bold: bool = False,
    size_pt: int = 11,
    font_cs: str = "",
    font_latin: str = "",
    width_in: float = 0.0,
    height_in: float = 0.0,
) -> str:
    if width_in and height_in:
        size_pt = _autofit_size_pt(text, width_in, height_in, size_pt)
    rfonts = ""
    if font_cs or font_latin:
        rfonts = f'<w:rFonts w:ascii="{font_latin}" w:hAnsi="{font_latin}" w:cs="{font_cs}"/>'
    bold_xml = "<w:b/><w:bCs/>" if bold else ""
    # No <w:rtl/> on the run, matching the flow renderer (docx.py), which
    # never sets it either — only <w:bidi/> on the paragraph. NOTE: tested
    # both ways via LibreOffice rendering and this alone did NOT fix the
    # bidi-reordering-on-wrap issue with mixed Persian/English text in a
    # narrow box (still under investigation — see conversation/plan notes).
    # Kept anyway since it matches the known-good renderer and cannot make
    # things worse.
    return (
        '<w:p><w:pPr><w:bidi/><w:jc w:val="right"/></w:pPr>'
        f'<w:r><w:rPr>{rfonts}{bold_xml}<w:sz w:val="{size_pt * 2}"/>'
        f'<w:szCs w:val="{size_pt * 2}"/></w:rPr>'
        f'<w:t xml:space="preserve">{_escape(text)}</w:t></w:r></w:p>'
    )


def _table_xml(node: ASTNode) -> str:
    if not node.rows:
        return ""
    cols = max(len(r) for r in node.rows)
    grid = "".join("<w:gridCol/>" for _ in range(cols))
    rows_xml = []
    for row in node.rows:
        cells = []
        for ci in range(cols):
            text = _escape(str(row[ci])) if ci < len(row) else ""
            cells.append(
                "<w:tc><w:tcPr><w:tcBorders>"
                '<w:top w:val="single" w:sz="4" w:color="999999"/>'
                '<w:bottom w:val="single" w:sz="4" w:color="999999"/>'
                '<w:left w:val="single" w:sz="4" w:color="999999"/>'
                '<w:right w:val="single" w:sz="4" w:color="999999"/>'
                "</w:tcBorders></w:tcPr>"
                f'<w:p><w:pPr><w:bidi/><w:jc w:val="right"/></w:pPr>'
                f'<w:r><w:rPr><w:sz w:val="18"/></w:rPr>'
                f'<w:t xml:space="preserve">{text}</w:t></w:r></w:p></w:tc>'
            )
        rows_xml.append(f'<w:tr>{"".join(cells)}</w:tr>')
    return f'<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(rows_xml)}</w:tbl>'


def _formula_xml(latex: str, font_cs: str, font_latin: str) -> str:
    try:
        omath = latex_to_omml(latex)
        return f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{omath}</w:p>'
    except LatexConversionError:
        logger.warning("Formula LaTeX->OMML conversion failed, using text fallback: %r", latex)
        rfonts = f'<w:rFonts w:ascii="{FONT_MATH}" w:hAnsi="{FONT_MATH}"/>'
        return (
            '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
            f'<w:r><w:rPr>{rfonts}<w:i/></w:rPr>'
            f'<w:t xml:space="preserve">{_escape(latex)}</w:t></w:r></w:p>'
        )


def _textbox_shape_xml(
    bbox: tuple[float, float, float, float], page: PageAST, inner_xml: str, include_shapetype: bool
):
    left, top, width, height = _bbox_to_inches(bbox, page)
    shapetype = _SHAPE_TYPE_XML if include_shapetype else ""
    xml = (
        f'<w:p {nsdecls("w")} {_VML_NS} {_O_NS} {_R_NS} {_M_NS}>'
        "<w:r><w:pict>"
        f"{shapetype}"
        f'<v:shape type="#_x0000_t202_di" style="position:absolute;'
        f"left:{left:.3f}in;top:{top:.3f}in;width:{width:.3f}in;height:{height:.3f}in;"
        'mso-position-horizontal-relative:page;mso-position-vertical-relative:page" '
        'filled="f" stroked="f">'
        '<v:textbox style="mso-fit-shape-to-text:t" inset="1pt,1pt,1pt,1pt">'
        f"<w:txbxContent>{inner_xml}</w:txbxContent>"
        "</v:textbox></v:shape></w:pict></w:r></w:p>"
    )
    return parse_xml(xml)


def _image_shape_xml(doc: Document, node: ASTNode, page: PageAST, include_shapetype: bool):
    asset_path = node.asset_path
    if not asset_path or not Path(asset_path).exists():
        return None
    try:
        with open(asset_path, "rb") as f:
            rid, _image = doc.part.get_or_add_image(BytesIO(f.read()))
    except Exception:
        logger.exception("Failed to embed image %s", asset_path)
        return None

    left, top, width, height = _bbox_to_inches(node.bbox, page)
    shapetype = _SHAPE_TYPE_XML if include_shapetype else ""
    xml = (
        f'<w:p {nsdecls("w")} {_VML_NS} {_O_NS} {_R_NS}>'
        "<w:r><w:pict>"
        f"{shapetype}"
        f'<v:shape style="position:absolute;left:{left:.3f}in;top:{top:.3f}in;'
        f"width:{width:.3f}in;height:{height:.3f}in;"
        'mso-position-horizontal-relative:page;mso-position-vertical-relative:page" '
        'filled="f" stroked="f">'
        f'<v:imagedata r:id="{rid}" o:title=""/>'
        "</v:shape></w:pict></w:r></w:p>"
    )
    return parse_xml(xml)
