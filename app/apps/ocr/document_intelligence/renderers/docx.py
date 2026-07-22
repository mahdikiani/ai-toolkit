"""Word Renderer — direct DOCX from Document AST with proper objects."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

from ..ast import DocumentAST, PageAST
from ..latex_omml import LatexConversionError, latex_to_omml
from ..layout import LayoutType

logger = logging.getLogger(__name__)

FONT_LATIN = "Calibri"
FONT_CS = "B Nazanin"
FONT_MATH = "Cambria Math"

# Page size/margins fall back to these if the source page dims are missing
# or implausible (e.g. a non-PDF image source with no natural "page").
DEFAULT_PAGE_WIDTH_IN = 8.27  # A4
DEFAULT_PAGE_HEIGHT_IN = 11.69
MARGIN_RATIO = 0.08
MIN_MARGIN_IN = 0.4
MAX_MARGIN_IN = 1.0

# How close a node's bbox center must be to the page center, and how narrow
# it must be, to be treated as visually centered rather than a right-aligned
# block that merely doesn't reach the page edge.
CENTER_TOLERANCE_RATIO = 0.08
CENTER_MAX_WIDTH_RATIO = 0.7

MIN_IMAGE_WIDTH_IN = 1.5


def render_docx(ast: DocumentAST, pdf_data: bytes | None = None) -> BytesIO:
    """Render full DocumentAST to a .docx BytesIO buffer.

    ``pdf_data`` (the original PDF bytes, when the source was a PDF) is used
    to detect the document's actual fonts instead of falling back to fixed
    Calibri/B Nazanin — see ``detect_pdf_fonts`` in the legacy pipeline
    renderer, reused here rather than reimplemented.
    """
    font_cs, font_latin = _resolve_fonts(pdf_data)
    page_width_in, page_height_in = _resolve_page_size(ast)
    margin_in = _resolve_margin(page_width_in)
    content_width_in = max(1.0, page_width_in - 2 * margin_in)

    doc = Document()
    _setup_styles(doc, font_cs, font_latin)
    _setup_default_section(doc, page_width_in, page_height_in, margin_in)

    if ast.title:
        doc.core_properties.title = ast.title

    header_text, footer_text = _collect_header_footer(ast)
    if header_text:
        _set_section_header(doc, header_text)
    if footer_text:
        _set_section_footer(doc, footer_text)

    prev_page = 0
    for page in ast.pages:
        if prev_page > 0 and page.page_number > prev_page:
            doc.add_page_break()
        _render_page(doc, page, content_width_in)
        prev_page = page.page_number

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def _resolve_fonts(pdf_data: bytes | None) -> tuple[str, str]:
    """Detect the source PDF's actual fonts; fall back to fixed defaults."""
    if not pdf_data:
        return FONT_CS, FONT_LATIN
    try:
        from ...pipeline.docx_renderer import detect_pdf_fonts

        detected = detect_pdf_fonts(pdf_data)
    except Exception:
        logger.debug("Font detection failed, using defaults", exc_info=True)
        return FONT_CS, FONT_LATIN
    return detected.get("cs", FONT_CS), detected.get("latin", FONT_LATIN)


def _resolve_page_size(ast: DocumentAST) -> tuple[float, float]:
    """Physical page size in inches, from the first page's rendered pixel
    dimensions at its render DPI. Falls back to A4 if unavailable."""
    for page in ast.pages:
        if page.page_width > 0 and page.page_height > 0 and page.page_dpi > 0:
            return (
                page.page_width / page.page_dpi,
                page.page_height / page.page_dpi,
            )
    return DEFAULT_PAGE_WIDTH_IN, DEFAULT_PAGE_HEIGHT_IN


def _resolve_margin(page_width_in: float) -> float:
    return max(MIN_MARGIN_IN, min(MAX_MARGIN_IN, page_width_in * MARGIN_RATIO))


def _collect_header_footer(ast: DocumentAST) -> tuple[str, str]:
    """Pick the most common header/footer text across pages (a repeating banner)."""
    from collections import Counter

    headers = [
        node.text.strip()
        for page in ast.pages
        for node in page.nodes
        if node.type == LayoutType.header and node.text.strip()
    ]
    footers = [
        node.text.strip()
        for page in ast.pages
        for node in page.nodes
        if node.type == LayoutType.footer and node.text.strip()
    ]
    header = Counter(headers).most_common(1)[0][0] if headers else ""
    footer = Counter(footers).most_common(1)[0][0] if footers else ""
    return header, footer


def _set_section_header(doc: Document, text: str) -> None:
    section = doc.sections[0]
    section.header.is_linked_to_previous = False
    p = section.header.paragraphs[0] if section.header.paragraphs else section.header.add_paragraph()
    p.text = text
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _ensure_rtl(p)


def _set_section_footer(doc: Document, text: str) -> None:
    section = doc.sections[0]
    section.footer.is_linked_to_previous = False
    p = section.footer.paragraphs[0] if section.footer.paragraphs else section.footer.add_paragraph()
    p.text = text
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _ensure_rtl(p)


def _setup_styles(doc: Document, font_cs: str, font_latin: str) -> None:
    style = doc.styles["Normal"]
    pf = style.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    def _xml(tag: str, **attrs: str) -> OxmlElement:
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        return el

    rpr = style.element.get_or_add_rPr()
    rFonts = rpr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = _xml("w:rFonts")
        rpr.append(rFonts)
    rFonts.set(qn("w:ascii"), font_latin)
    rFonts.set(qn("w:hAnsi"), font_latin)
    rFonts.set(qn("w:cs"), font_cs)
    rFonts.set(qn("w:eastAsia"), font_latin)

    pPr = style.element.get_or_add_pPr()
    if pPr.find(qn("w:bidi")) is None:
        pPr.append(_xml("w:bidi"))
    sz = rpr.find(qn("w:sz"))
    if sz is None:
        rpr.append(_xml("w:sz", **{"w:val": "22"}))

    for level in range(1, 4):
        hs = doc.styles[f"Heading {level}"]
        hs.font.bold = True
        hs.font.color.rgb = RGBColor(0, 0, 0)
        hpf = hs.paragraph_format
        hpf.space_before = Pt(12)
        hpf.space_after = Pt(6)
        hpf.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _set_bidi_and_fonts(hs, font_cs, font_latin)


def _set_bidi_and_fonts(style, font_cs: str, font_latin: str) -> None:
    def _xml(tag: str, **attrs: str) -> OxmlElement:
        el = OxmlElement(tag)
        for k, v in attrs.items():
            el.set(qn(k), v)
        return el

    hpr = style.element.get_or_add_rPr()
    if hpr.find(qn("w:bidi")) is None:
        hpr.append(_xml("w:bidi"))
    hrFonts = hpr.find(qn("w:rFonts"))
    if hrFonts is None:
        hrFonts = _xml("w:rFonts")
        hpr.append(hrFonts)
    hrFonts.set(qn("w:cs"), font_cs)
    hrFonts.set(qn("w:ascii"), font_latin)
    hrFonts.set(qn("w:hAnsi"), font_latin)


def _setup_default_section(
    doc: Document, page_width_in: float, page_height_in: float, margin_in: float
) -> None:
    for section in doc.sections:
        section.page_width = Inches(page_width_in)
        section.page_height = Inches(page_height_in)
        section.top_margin = Inches(margin_in)
        section.bottom_margin = Inches(margin_in)
        section.left_margin = Inches(margin_in)
        section.right_margin = Inches(margin_in)


def _render_page(doc: Document, page: PageAST, content_width_in: float) -> None:
    for node in page.nodes:
        _render_node(doc, node, page.page_width, content_width_in)


def _render_node(doc: Document, node, page_width_px: float, content_width_in: float) -> None:
    if node.type == LayoutType.title:
        _add_heading(doc, node.text, 1, node, page_width_px)
    elif node.type == LayoutType.heading:
        _add_heading(doc, node.text, min(node.level + 1, 3), node, page_width_px)
    elif node.type in (LayoutType.header, LayoutType.footer):
        # Promoted to a real Word header/footer section — see _collect_header_footer.
        return
    elif node.type == LayoutType.paragraph:
        _add_paragraph(doc, node.text, node=node, page_width_px=page_width_px)
    elif node.type == LayoutType.reference:
        _add_paragraph(doc, f"📎 {node.text}", node=node, page_width_px=page_width_px)
    elif node.type == LayoutType.list:
        _add_list(doc, node)
    elif node.type == LayoutType.table:
        _add_table(doc, node)
    elif node.type == LayoutType.formula:
        _add_formula(doc, node.latex)
    elif node.type == LayoutType.figure:
        _add_image(doc, node, page_width_px, content_width_in)
    elif node.type == LayoutType.chart:
        _add_chart(doc, node, page_width_px, content_width_in)
    elif node.type == LayoutType.code:
        _add_code(doc, node)


def _resolve_alignment(node, page_width_px: float):
    """Right-align by default (RTL body text); promote to CENTER when the
    node's bbox is genuinely centered on the page rather than merely
    falling short of the page edge (e.g. a centered title/box heading)."""
    if not page_width_px or node.bbox == (0.0, 0.0, 0.0, 0.0):
        return WD_ALIGN_PARAGRAPH.RIGHT
    x1, _, x2, _ = node.bbox
    width = x2 - x1
    center_offset = abs((x1 + x2) / 2 - page_width_px / 2)
    if (
        width < page_width_px * CENTER_MAX_WIDTH_RATIO
        and center_offset < page_width_px * CENTER_TOLERANCE_RATIO
    ):
        return WD_ALIGN_PARAGRAPH.CENTER
    return WD_ALIGN_PARAGRAPH.RIGHT


def _add_heading(doc: Document, text: str, level: int, node=None, page_width_px: float = 0.0) -> None:
    level = max(1, min(level, 3))
    p = doc.add_heading(text, level=level)
    p.alignment = _resolve_alignment(node, page_width_px) if node else WD_ALIGN_PARAGRAPH.RIGHT
    _ensure_rtl(p)


def _add_paragraph(
    doc: Document, text: str, italic: bool = False, node=None, page_width_px: float = 0.0
) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text or "")
    run.italic = italic
    p.alignment = _resolve_alignment(node, page_width_px) if node else WD_ALIGN_PARAGRAPH.RIGHT
    _ensure_rtl(p)


def _add_list(doc: Document, node) -> None:
    for child in node.children:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(child.text or "")
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT


def _add_table(doc: Document, node) -> None:
    if not node.rows:
        return
    rows = node.rows
    cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    table.style = "Table Grid"
    for ri, row_data in enumerate(rows):
        for ci, cell_text in enumerate(row_data):
            if ci >= cols:
                break
            cell = table.cell(ri, ci)
            cell.text = str(cell_text)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                _ensure_rtl(paragraph)


def _add_formula(doc: Document, latex: str) -> None:
    """
    Add formula as a real, editable OMML (Office Math) Word equation.

    LaTeX is parsed and re-emitted as genuine ``m:oMath`` XML (see
    ../latex_omml.py) so Word treats it as an equation object, not text. If
    the LaTeX can't be parsed, fall back to styled math-font text instead of
    failing the whole document.
    """
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    try:
        omath_xml = latex_to_omml(latex)
        oMathPara = parse_xml(f'<m:oMathPara {nsdecls("m")}>{omath_xml}</m:oMathPara>')
        p._p.append(oMathPara)
    except LatexConversionError:
        logger.warning("Formula LaTeX->OMML conversion failed, using text fallback: %r", latex)
        run = p.add_run(latex)
        run.italic = True
        run.font.name = FONT_MATH


def _resolve_image_width_in(node, page_width_px: float, content_width_in: float) -> float:
    """Size the image relative to how much of the page width its original
    bbox occupied, instead of always inserting a fixed 5 inches — so a small
    inline figure doesn't dominate the page and a full-width diagram isn't
    shrunk down to match everything else."""
    if not page_width_px or node.bbox == (0.0, 0.0, 0.0, 0.0):
        return min(5.0, content_width_in)
    x1, _, x2, _ = node.bbox
    ratio = max(0.0, min(1.0, (x2 - x1) / page_width_px))
    width_in = ratio * content_width_in
    return max(MIN_IMAGE_WIDTH_IN, min(width_in, content_width_in))


def _add_image(
    doc: Document, node, page_width_px: float = 0.0, content_width_in: float = 6.0
) -> None:
    asset_path = node.asset_path
    if not asset_path or not Path(asset_path).exists():
        _add_paragraph(doc, f"[تصویر: {node.caption}]", italic=True)
        return

    try:
        img_bytes = Path(asset_path).read_bytes()
        width_in = _resolve_image_width_in(node, page_width_px, content_width_in)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(BytesIO(img_bytes), width=Inches(width_in))
        if node.caption:
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap_run = cap.add_run(node.caption)
            cap_run.italic = True
    except Exception:
        _add_paragraph(doc, f"[تصویر: {node.caption}]", italic=True)


def _add_chart(
    doc: Document, node, page_width_px: float = 0.0, content_width_in: float = 6.0
) -> None:
    """Render chart — image + caption + optional data table."""
    _add_image(doc, node, page_width_px, content_width_in)
    if node.caption:
        _add_paragraph(doc, node.caption, italic=True)
    if node.description:
        _add_paragraph(doc, node.description)
    if node.chart_data and node.chart_data.get("data"):
        rows = [["Label", "Value"]]
        for item in node.chart_data["data"]:
            if isinstance(item, dict):
                rows.append([str(item.get("label", "")), str(item.get("value", ""))])
        tbl = doc.add_table(rows=len(rows), cols=2)
        tbl.style = "Table Grid"
        for ri, row_data in enumerate(rows):
            for ci, val in enumerate(row_data):
                tbl.cell(ri, ci).text = val


def _add_code(doc: Document, node) -> None:
    p = doc.add_paragraph()
    run = p.add_run(node.text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    _set_shading(p)


def _set_shading(p) -> None:
    pPr = p._element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F0F0F0")
    pPr.append(shd)


def _ensure_rtl(p) -> None:
    pPr = p._element.get_or_add_pPr()
    if pPr.find(qn("w:bidi")) is None:
        pPr.append(OxmlElement("w:bidi"))
