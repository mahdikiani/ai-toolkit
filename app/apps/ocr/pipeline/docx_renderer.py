"""Deterministic DOCX renderer — builds a Word document from pipeline output."""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image as PILImage

from .layout_detector import ElementType, LayoutBox

logger = logging.getLogger(__name__)

FONT_PERSIAN = "B Nazanin"
FONT_LATIN = "Calibri"
FONT_CS = "B Nazanin"  # Complex-script (Persian/Arabic)


def detect_pdf_fonts(pdf_data: bytes | None = None) -> dict[str, str]:
    """Extract font names from PDF bytes.

    Returns the most common complex-script (Persian/Arabic) and latin-script font.
    """
    if not pdf_data:
        return {}
    import fitz

    latin_fonts: dict[str, int] = {}
    cs_fonts: dict[str, int] = {}
    try:
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        for page in doc:
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        name = span.get("font", "")
                        if not name:
                            continue
                        is_cs = bool(span.get("bidi") or 0) or any(
                            ord(c) > 0x0600 for c in span.get("text", "")[:10]
                        )
                        if is_cs:
                            cs_fonts[name] = cs_fonts.get(name, 0) + 1
                        else:
                            latin_fonts[name] = latin_fonts.get(name, 0) + 1
        doc.close()
    except Exception:
        logger.debug("Could not read fonts from PDF", exc_info=True)

    result: dict[str, str] = {}
    if cs_fonts:
        result["cs"] = max(cs_fonts, key=cs_fonts.get)
    if latin_fonts:
        result["latin"] = max(latin_fonts, key=latin_fonts.get)
    return result


def build_docx(
    markdown: str,
    page_images: list[PILImage.Image],
    elements: list[LayoutBox] | None = None,
    crops_dir: str | Path | None = None,
    assets_dir: str | Path | None = None,
    pdf_data: bytes | None = None,
) -> BytesIO:
    """Convert pipeline Markdown output to a .docx BytesIO.

    The Markdown is parsed into blocks (headings, paragraphs, images, tables,
    formulas, separators).  Page images are used to crop out figure/chart
    regions when actual image assets are unavailable.
    """
    doc = Document()

    detected = detect_pdf_fonts(pdf_data)
    cs_font = detected.get("cs") or FONT_PERSIAN
    latin_font = detected.get("latin") or FONT_LATIN
    logger.info("DOCX fonts — Persian: %s, English: %s", cs_font, latin_font)

    _setup_rtl_styles(doc, cs_font=cs_font, latin_font=latin_font)

    blocks = _parse_markdown_blocks(markdown)

    for block in blocks:
        kind = block.get("kind")
        if kind == "heading":
            _add_heading(doc, block["text"], block["level"])
        elif kind == "paragraph":
            _add_paragraph(doc, block["text"])
        elif kind == "list_item":
            _add_paragraph(doc, f"• {block['text']}")
        elif kind == "image":
            img_bytes = _resolve_image(block, page_images, elements, crops_dir, assets_dir)
            if img_bytes:
                _add_image(doc, img_bytes, block.get("alt", ""))
        elif kind == "table":
            _add_table(doc, block)
        elif kind == "formula":
            _add_formula(doc, block["latex"], block.get("display", False))
        elif kind == "separator":
            doc.add_page_break()
        elif kind == "blockquote":
            _add_paragraph(doc, block["text"], italic=True)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


def _setup_rtl_styles(
    doc: Document,
    *,
    cs_font: str = FONT_PERSIAN,
    latin_font: str = FONT_LATIN,
) -> None:
    """Configure default RTL text direction and Persian-friendly fonts."""
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
    rFonts.set(qn("w:ascii"), latin_font)
    rFonts.set(qn("w:hAnsi"), latin_font)
    rFonts.set(qn("w:cs"), cs_font)

    pPr = style.element.get_or_add_pPr()
    bidi = pPr.find(qn("w:bidi"))
    if bidi is None:
        pPr.append(_xml("w:bidi"))

    sz = rpr.find(qn("w:sz"))
    if sz is None:
        rpr.append(_xml("w:sz", **{"w:val": "22"}))

    for level in range(1, 5):
        hs = doc.styles[f"Heading {level}"]
        hs.font.name = cs_font
        hs.font.bold = True
        hs.font.color.rgb = RGBColor(0, 0, 0)
        pf = hs.paragraph_format
        pf.space_before = Pt(12)
        pf.space_after = Pt(6)
        pf.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hpr = hs.element.get_or_add_rPr()
        if hpr.find(qn("w:bidi")) is None:
            hpr.append(_xml("w:bidi"))
        hrFonts = hpr.find(qn("w:rFonts"))
        if hrFonts is None:
            hrFonts = _xml("w:rFonts")
            hpr.append(hrFonts)
        hrFonts.set(qn("w:cs"), cs_font)
        hrFonts.set(qn("w:ascii"), latin_font)
        hrFonts.set(qn("w:hAnsi"), latin_font)


def _add_heading(doc: Document, text: str, level: int) -> None:
    level = max(1, min(level, 4))
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _ensure_rtl(p)


def _add_paragraph(doc: Document, text: str, italic: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _ensure_rtl(p)


def _add_image(doc: Document, img_bytes: bytes, alt: str = "") -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(BytesIO(img_bytes), width=Inches(5.0))
    if alt:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap_run = cap.add_run(alt)
        cap_run.italic = True


def _add_table(doc: Document, block: dict) -> None:
    rows_data = block.get("rows", [])
    if not rows_data:
        return
    cols = max(len(r) for r in rows_data)
    table = doc.add_table(rows=len(rows_data), cols=cols)
    table.style = "Table Grid"
    for ri, row_data in enumerate(rows_data):
        for ci, cell_text in enumerate(row_data):
            if ci >= cols:
                break
            cell = table.cell(ri, ci)
            cell.text = str(cell_text)
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                _ensure_rtl(paragraph)


def _add_formula(doc: Document, latex: str, display: bool) -> None:
    text = f"$${latex}$$" if display else f"${latex}$"
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _ensure_rtl(p) -> None:
    pPr = p._element.get_or_add_pPr()
    bidi = pPr.find(qn("w:bidi"))
    if bidi is None:
        el = OxmlElement("w:bidi")
        pPr.append(el)


def _parse_markdown_blocks(md: str) -> list[dict]:
    """Parse Markdown into a list of typed blocks."""
    blocks: list[dict] = []
    for line in md.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("<!-- page:"):
            blocks.append({"kind": "separator"})
            continue

        if stripped == "---":
            blocks.append({"kind": "separator"})
            continue

        heading_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2)
            blocks.append({"kind": "heading", "level": level, "text": text})
            continue

        img_match = re.match(r"!\[(.+?)\]\((.+?)\)", stripped)
        if img_match:
            blocks.append({"kind": "image", "alt": img_match.group(1), "src": img_match.group(2)})
            continue

        table_match = re.match(r"^\|(.+)\|$", stripped)
        if table_match:
            cells = [c.strip() for c in table_match.group(1).split("|")]
            if blocks and blocks[-1].get("kind") == "table":
                blocks[-1].setdefault("rows", []).append(cells)
            else:
                blocks.append({"kind": "table", "rows": [cells]})
            continue

        table_sep = re.match(r"^\|[-:| ]+\|$", stripped)
        if table_sep:
            continue

        formula_match = re.match(r"^\$\$(.+)\$\$$", stripped)
        if formula_match:
            blocks.append({"kind": "formula", "latex": formula_match.group(1), "display": True})
            continue

        inline_formula = re.match(r"^\$(.+)\$$", stripped)
        if inline_formula:
            blocks.append({"kind": "formula", "latex": inline_formula.group(1), "display": False})
            continue

        if stripped.startswith("> "):
            blocks.append({"kind": "blockquote", "text": stripped[2:]})
            continue

        if re.match(r"^[\-\*]\s", stripped):
            blocks.append({"kind": "list_item", "text": re.sub(r"^[\-\*]\s+", "", stripped)})
            continue

        if re.match(r"^\d+[.\)]\s", stripped):
            blocks.append({"kind": "list_item", "text": re.sub(r"^\d+[.\)]\s+", "", stripped)})
            continue

        blocks.append({"kind": "paragraph", "text": stripped})

    return blocks


def _resolve_image(
    block: dict,
    page_images: list[PILImage.Image],
    elements: list[LayoutBox] | None,
    crops_dir: str | Path | None,
    assets_dir: str | Path | None,
) -> bytes | None:
    """Try to locate the actual image bytes for an image block."""
    src = block.get("src", "")
    if src and src != "#" and not src.startswith("asset:"):
        path = Path(src)
        if path.exists():
            return path.read_bytes()
        if assets_dir:
            alt_path = Path(assets_dir) / path.name
            if alt_path.exists():
                return alt_path.read_bytes()

    alt = block.get("alt", "")
    page_num = block.get("_page", 1)

    if elements and page_images:
        for elem in elements:
            if elem.type in (ElementType.figure, ElementType.chart):
                if page_images:
                    img = page_images[min(elem.page_number - 1, len(page_images) - 1)]
                    x1 = max(0, int(elem.x1) - 5)
                    y1 = max(0, int(elem.y1) - 5)
                    x2 = min(img.width, int(elem.x2) + 5)
                    y2 = min(img.height, int(elem.y2) + 5)
                    if x2 > x1 and y2 > y1:
                        crop = img.crop((x1, y1, x2, y2))
                        buf = BytesIO()
                        crop.save(buf, format="PNG")
                        buf.seek(0)
                        return buf.read()

    if crops_dir:
        for f in Path(crops_dir).iterdir():
            if f.suffix in (".png", ".jpg", ".jpeg"):
                return f.read_bytes()

    return None
