"""markdown → pdf conversion strategy (WeasyPrint via Document Intelligence)."""

from __future__ import annotations

from apps.artifacts.enums import ArtifactFormat
from apps.converter.registry import register
from apps.ocr.document_intelligence.markdown_parser import parse_markdown
from apps.ocr.document_intelligence.renderers.pdf import render_pdf


def markdown_bytes_to_pdf(source_bytes: bytes, *, title: str = "") -> bytes:
    """Convert UTF-8 markdown bytes to a PDF document."""
    markdown = source_bytes.decode("utf-8", errors="replace")
    return markdown_text_to_pdf(markdown, title=title)


def markdown_text_to_pdf(markdown: str, *, title: str = "") -> bytes:
    """Convert a markdown string to PDF bytes via the shared DI renderer."""
    ast = parse_markdown(markdown, title=title)
    return render_pdf(ast).getvalue()


register(
    ArtifactFormat.markdown,
    ArtifactFormat.pdf,
    markdown_bytes_to_pdf,
    name="markdown_pdf",
)
