"""markdown → docx conversion strategy (Document Intelligence renderers)."""

from __future__ import annotations

from apps.artifacts.enums import ArtifactFormat
from apps.converter.registry import register
from apps.ocr.document_intelligence.markdown_parser import parse_markdown
from apps.ocr.document_intelligence.renderers.docx import render_docx


def markdown_bytes_to_docx(source_bytes: bytes, *, title: str = "") -> bytes:
    """Convert UTF-8 markdown bytes to a DOCX document."""
    markdown = source_bytes.decode("utf-8", errors="replace")
    return markdown_text_to_docx(markdown, title=title)


def markdown_text_to_docx(markdown: str, *, title: str = "") -> bytes:
    """Convert a markdown string to DOCX bytes via the shared DI renderer."""
    ast = parse_markdown(markdown, title=title)
    return render_docx(ast).getvalue()


register(
    ArtifactFormat.markdown,
    ArtifactFormat.docx,
    markdown_bytes_to_docx,
    name="markdown_docx",
)
