"""
Synchronous Markdown document conversion API.

Compatibility shim for mirza-bot and other clients that still POST markdown
and expect a streamed DOCX/PDF download. Rendering is owned by
``apps.converter`` strategies (same registry edges as Artifact conversion);
this module only adapts request shape → StreamingResponse.

Prefer ``POST /convert`` with an ``artifact_id`` for new integrations.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from usso import UserData

from apps.artifacts.enums import ArtifactFormat
from apps.converter.services import render_markdown_to_format
from utils.usso import get_usso

router = APIRouter(prefix="/document-convert", tags=["Document Convert"])
auth = get_usso(raise_exception=True)

_DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
_PDF_MEDIA_TYPE = "application/pdf"


class MarkdownToDocxRequest(BaseModel):
    """Request body for Markdown -> DOCX/PDF conversion."""

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


def _docx_response(markdown: str, title: str, file_name: str) -> StreamingResponse:
    body = render_markdown_to_format(
        markdown, target_format=ArtifactFormat.docx, title=title
    )
    return StreamingResponse(
        BytesIO(body),
        media_type=_DOCX_MEDIA_TYPE,
        headers={"Content-Disposition": _content_disposition(file_name)},
    )


def _pdf_response(markdown: str, title: str, file_name: str) -> StreamingResponse:
    body = render_markdown_to_format(
        markdown, target_format=ArtifactFormat.pdf, title=title
    )
    return StreamingResponse(
        BytesIO(body),
        media_type=_PDF_MEDIA_TYPE,
        headers={"Content-Disposition": _content_disposition(file_name)},
    )


@router.post("/markdown-to-docx")
async def markdown_to_docx(
    data: MarkdownToDocxRequest,
    user: UserData = Depends(auth),
) -> StreamingResponse:
    """Convert a Markdown string to a real, RTL-correct .docx file."""
    del user
    return _docx_response(data.markdown, data.title, "document.docx")


@router.post("/markdown-to-docx/upload")
async def markdown_to_docx_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    user: UserData = Depends(auth),
) -> StreamingResponse:
    """Convert an uploaded .md file to a real, RTL-correct .docx file."""
    del user
    raw = await file.read()
    markdown = raw.decode("utf-8", errors="replace")
    resolved_title = title or (Path(file.filename).stem if file.filename else "")
    return _docx_response(
        markdown, resolved_title, f"{resolved_title or 'document'}.docx"
    )


@router.post("/markdown-to-pdf")
async def markdown_to_pdf(
    data: MarkdownToDocxRequest,
    user: UserData = Depends(auth),
) -> StreamingResponse:
    """Convert a Markdown string to an A4, RTL-correct PDF file."""
    del user
    return _pdf_response(data.markdown, data.title, "document.pdf")


@router.post("/markdown-to-pdf/upload")
async def markdown_to_pdf_upload(
    file: UploadFile = File(...),
    title: str = Form(""),
    user: UserData = Depends(auth),
) -> StreamingResponse:
    """Convert an uploaded .md file to an A4, RTL-correct PDF file."""
    del user
    raw = await file.read()
    markdown = raw.decode("utf-8", errors="replace")
    resolved_title = title or (Path(file.filename).stem if file.filename else "")
    return _pdf_response(
        markdown, resolved_title, f"{resolved_title or 'document'}.pdf"
    )
