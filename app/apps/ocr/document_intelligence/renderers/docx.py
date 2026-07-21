"""Word Renderer — direct DOCX from Document AST with proper objects."""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, qn
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor, Emu

from .ast import DocumentAST, PageAST
from .layout import LayoutType

logger = logging.getLogger(__name__)

FONT_LATIN = "Calibri"
FONT_CS = "B Nazanin"
FONT_MATH = "Cambria Math"


def render_docx(ast: DocumentAST) -> BytesIO:
    """Render full DocumentAST to a .docx BytesIO buffer."""
    doc = Document()
    _setup_styles(doc)
    _setup_default_section(doc)

    prev_page = 0
    for page in ast.pages:
        if prev_page > 0 and page.page_number > prev_page:
            doc.add_page_break()
        _render_page(doc, page)
        prev_page = page.page_number

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def _setup_styles(doc: Document) -> None:
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
    rFonts.set(qn("w:ascii"), FONT_LATIN)
    rFonts.set(qn("w:hAnsi"), FONT_LATIN)
    rFonts.set(qn("w:cs"), FONT_CS)
    rFonts.set(qn("w:eastAsia"), FONT_LATIN)

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
        _set_bidi_and_fonts(hs)


def _set_bidi_and_fonts(style) -> None:
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
    hrFonts.set(qn("w:cs"), FONT_CS)
    hrFonts.set(qn("w:ascii"), FONT_LATIN)
    hrFonts.set(qn("w:hAnsi"), FONT_LATIN)


def _setup_default_section(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)


def _render_page(doc: Document, page: PageAST) -> None:
    for node in page.nodes:
        _render_node(doc, node)


def _render_node(doc: Document, node) -> None:
    if node.type == LayoutType.title:
        _add_heading(doc, node.text, 1)
    elif node.type == LayoutType.heading:
        _add_heading(doc, node.text, min(node.level + 1, 3))
    elif node.type in (LayoutType.header, LayoutType.footer):
        _add_paragraph(doc, node.text, italic=True)
    elif node.type == LayoutType.paragraph:
        _add_paragraph(doc, node.text)
    elif node.type == LayoutType.reference:
        _add_paragraph(doc, f"📎 {node.text}")
    elif node.type == LayoutType.list:
        _add_list(doc, node)
    elif node.type == LayoutType.table:
        _add_table(doc, node)
    elif node.type == LayoutType.formula:
        _add_formula(doc, node.latex)
    elif node.type == LayoutType.figure:
        _add_image(doc, node)
    elif node.type == LayoutType.chart:
        _add_chart(doc, node)
    elif node.type == LayoutType.code:
        _add_code(doc, node)


def _add_heading(doc: Document, text: str, level: int) -> None:
    level = max(1, min(level, 3))
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _ensure_rtl(p)


def _add_paragraph(doc: Document, text: str, italic: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text or "")
    run.italic = italic
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _ensure_rtl(p)


def _add_list(doc: Document, node) -> None:
    for child in node.children:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(child.text or "")
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
    """Add formula as proper OMML (Office Math) equation."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # OMML-based equation
    oMathPara = OxmlElement("m:oMathPara")
    oMathPara.set(qn("m:xmlns:m"), "http://schemas.openxmlformats.org/officeDocument/2006/math")
    oMath = OxmlElement("m:oMath")
    oMathPara.append(oMath)

    # Add LaTeX as text run inside equation
    r = OxmlElement("m:r")
    rPr = OxmlElement("m:rPr")
    rStyle = OxmlElement("m:sty")
    rStyle.set(qn("m:val"), "p")
    rPr.append(rStyle)
    r.append(rPr)
    t = OxmlElement("m:t")
    t.text = latex
    r.append(t)
    oMath.append(r)

    p._element.append(oMathPara)


def _add_image(doc: Document, node) -> None:
    asset_path = node.asset_path
    if not asset_path or not Path(asset_path).exists():
        _add_paragraph(doc, f"[تصویر: {node.caption}]", italic=True)
        return

    try:
        img_bytes = Path(asset_path).read_bytes()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(BytesIO(img_bytes), width=Inches(5.0))
        if node.caption:
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap_run = cap.add_run(node.caption)
            cap_run.italic = True
    except Exception:
        _add_paragraph(doc, f"[تصویر: {node.caption}]", italic=True)


def _add_chart(doc: Document, node) -> None:
    """Render chart — image + caption + optional data table."""
    _add_image(doc, node)
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
