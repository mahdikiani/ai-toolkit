"""PDF page rendering using PyMuPDF."""

import logging
from io import BytesIO
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def render_pdf_pages(pdf_path: str | Path, dpi: int = 300) -> list[Image.Image]:
    """Render all PDF pages to PIL Images using PyMuPDF (fitz)."""
    import fitz

    doc = fitz.open(str(pdf_path))
    pages: list[Image.Image] = []
    try:
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            rotation = page.rotation or 0
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom).prerotate(rotation)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
            logger.debug(
                "Rendered page %d: %dx%d @ %ddpi",
                page_number + 1,
                pix.width,
                pix.height,
                dpi,
            )
    finally:
        doc.close()
    logger.info("Rendered %d pages @ %d DPI", len(pages), dpi)
    return pages


def render_pdf_bytes(pdf_bytes: BytesIO, dpi: int = 300) -> list[Image.Image]:
    """Render PDF from bytes to PIL Images."""
    import fitz

    pdf_bytes.seek(0)
    data = pdf_bytes.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[Image.Image] = []
    try:
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            rotation = page.rotation or 0
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom).prerotate(rotation)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
    finally:
        doc.close()
    return pages


def count_pdf_bytes(pdf_bytes: BytesIO) -> int:
    """Return the exact PDF page count without rasterizing its pages."""
    import fitz

    pdf_bytes.seek(0)
    doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
    try:
        return len(doc)
    finally:
        doc.close()
