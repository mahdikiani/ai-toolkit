"""
Synchronous Markdown document conversion API.

Unlike the OCR/webpage/transcribe apps, this needs no task/polling model:
it's pure CPU text-to-XML work (no OCR/VLM/LLM calls), done in well under
a second, so a plain synchronous request/response is the right shape.

Exists so mirza-bot's "convert to Word" button can produce a real,
RTL-correct .docx (real tables, OMML formulas, per-run bold/italic,
correct w:lang bidi handling) through the same renderer the OCR pipeline
uses, instead of maintaining a separate, cruder pandoc-based conversion
locally.

Each output format has JSON-string and uploaded-file variants. Both reuse
the shared ``parse_markdown`` path and the corresponding DocumentAST renderer.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from usso import UserData

from utils.usso import get_usso

from .document_intelligence.ast import DocumentAST
from .document_intelligence.markdown_parser import parse_markdown
from .document_intelligence.renderers.docx import render_docx
from .document_intelligence.renderers.pdf import render_pdf

router = APIRouter(prefix="/document-convert", tags=["Document Convert"])
auth = get_usso(raise_exception=True)

_DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
_PDF_MEDIA_TYPE = "application/pdf"


class MarkdownToDocxRequest(BaseModel):
    """Request body for Markdown -> DOCX conversion."""

    markdown: str
    title: str = ""


def _content_disposition(file_name: str) -> str:
    """
    Build a Content-Disposition header that survives a non-ASCII file name.

    HTTP header values must be Latin-1 -- a title/filename straight from
    a Persian document would otherwise raise UnicodeEncodeError deep
    inside Starlette and turn a perfectly valid conversion into a 500.
    Per RFC 6266/5987: an ASCII-safe ``filename`` for clients that don't
    understand the extended form, plus the real name in ``filename*``.
    """
    ascii_fallback = file_name.encode("ascii", "ignore").decode("ascii").strip()
    ascii_fallback = ascii_fallback or "document.docx"
    encoded = quote(file_name)
    return f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded}'


def _docx_response(ast: DocumentAST, file_name: str) -> StreamingResponse:
    buf = render_docx(ast)
    return StreamingResponse(
        buf,
        media_type=_DOCX_MEDIA_TYPE,
        headers={"Content-Disposition": _content_disposition(file_name)},
    )


def _pdf_response(ast: DocumentAST, file_name: str) -> StreamingResponse:
    buf = render_pdf(ast)
    return StreamingResponse(
        buf,
        media_type=_PDF_MEDIA_TYPE,
        headers={"Content-Disposition": _content_disposition(file_name)},
    )


@router.post("/markdown-to-docx")
async def markdown_to_docx(
    data: MarkdownToDocxRequest,
    user: UserData = Depends(auth),
) -> StreamingResponse:
    """Convert a Markdown string to a real, RTL-correct .docx file."""
    ast = parse_markdown(data.markdown, title=data.title)
    return _docx_response(ast, "document.docx")


@router.post("/markdown-to-docx/upload")
async def markdown_to_docx_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    user: UserData = Depends(auth),
) -> StreamingResponse:
    """Convert an uploaded .md file to a real, RTL-correct .docx file."""
    raw = await file.read()
    markdown = raw.decode("utf-8", errors="replace")
    resolved_title = title or (Path(file.filename).stem if file.filename else "")
    ast = parse_markdown(markdown, title=resolved_title)
    return _docx_response(ast, f"{resolved_title or 'document'}.docx")


@router.post("/markdown-to-pdf")
async def markdown_to_pdf(
    data: MarkdownToDocxRequest,
    user: UserData = Depends(auth),
) -> StreamingResponse:
    """Convert a Markdown string to an A4, RTL-correct PDF file."""
    ast = parse_markdown(data.markdown, title=data.title)
    return _pdf_response(ast, "document.pdf")


@router.post("/markdown-to-pdf/upload")
async def markdown_to_pdf_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    user: UserData = Depends(auth),
) -> StreamingResponse:
    """Convert an uploaded .md file to an A4, RTL-correct PDF file."""
    raw = await file.read()
    markdown = raw.decode("utf-8", errors="replace")
    resolved_title = title or (Path(file.filename).stem if file.filename else "")
    ast = parse_markdown(markdown, title=resolved_title)
    return _pdf_response(ast, f"{resolved_title or 'document'}.pdf")
