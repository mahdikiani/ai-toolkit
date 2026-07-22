"""Document Loader — detects file type, extracts pages as images."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Page:
    id: str
    page_number: int
    image: Image.Image
    width: int
    height: int
    dpi: int = 300


@dataclass
class Document:
    id: str
    filename: str
    pages: list[Page]
    pages_count: int
    created_at: str
    pdf_data: bytes | None = None


def load_document(source: str | Path | BytesIO, dpi: int = 300) -> Document:
    """Load any supported document type and return a Document with rendered pages."""
    doc_id = str(uuid.uuid4())
    created = datetime.now(tz=UTC).isoformat()

    if isinstance(source, (str, Path)):
        path = Path(source) if isinstance(source, str) else source
        filename = path.name
        raw_bytes = path.read_bytes()
    elif isinstance(source, BytesIO):
        raw_bytes = source.read()
        source.seek(0)
        filename = getattr(source, "name", "unknown")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    mime = _detect_mime(raw_bytes, filename)

    if mime == "application/pdf":
        pages = _render_pdf(raw_bytes, dpi)
    elif mime.startswith("image/"):
        single = Image.open(BytesIO(raw_bytes))
        if single.mode != "RGB":
            single = single.convert("RGB")
        pages = [
            Page(
                id=f"{doc_id}_p001",
                page_number=1,
                image=single.copy(),
                width=single.width,
                height=single.height,
                dpi=dpi,
            )
        ]
    else:
        raise ValueError(f"Unsupported file type: {mime}")

    logger.info("Loaded %s — %d pages (%s)", filename, len(pages), mime)
    return Document(
        id=doc_id,
        filename=filename,
        pages=pages,
        pages_count=len(pages),
        created_at=created,
        pdf_data=raw_bytes if mime == "application/pdf" else None,
    )


def _detect_mime(data: bytes, filename: str = "") -> str:
    """Detect MIME type from bytes or filename."""

    if data[:4] == b"%PDF":
        return "application/pdf"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:2] in (b"BM",):
        return "image/bmp"
    if data[:4] == b"RIFF":
        return "image/webp"

    ext = Path(filename).suffix.lower() if filename else ""
    ext_map = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    return ext_map.get(ext, "application/octet-stream")


def _render_pdf(pdf_bytes: bytes, dpi: int) -> list[Page]:
    """Render all PDF pages to PIL Images."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    scale = float(dpi) / 72.0
    pages: list[Page] = []
    try:
        total = len(doc)
        for i in range(total):
            bitmap = doc[i].render(scale=scale)
            pil = bitmap.to_pil()
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            pages.append(
                Page(
                    id=f"p{i + 1:04d}",
                    page_number=i + 1,
                    image=pil,
                    width=pil.width,
                    height=pil.height,
                    dpi=dpi,
                )
            )
    finally:
        doc.close()
    return pages
