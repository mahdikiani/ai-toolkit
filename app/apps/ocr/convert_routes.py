"""
Synchronous Markdown -> DOCX conversion API.

Unlike the OCR/webpage/transcribe apps, this needs no task/polling model:
it's pure CPU text-to-XML work (no OCR/VLM/LLM calls), done in well under a
second, so a plain synchronous request/response is the right shape.

Exists so mirza-bot's "convert to Word" button can produce a real,
RTL-correct .docx (real tables, OMML formulas, per-run bold/italic, correct
w:lang bidi handling) through the same renderer the OCR pipeline uses,
instead of maintaining a separate, cruder pandoc-based conversion locally.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from usso import UserData

from utils.usso import get_usso

from .document_intelligence.markdown_parser import parse_markdown
from .document_intelligence.renderers.docx import render_docx

router = APIRouter(prefix="/document-convert", tags=["Document Convert"])
auth = get_usso(raise_exception=True)


class MarkdownToDocxRequest(BaseModel):
    """Request body for Markdown -> DOCX conversion."""

    markdown: str
    title: str = ""


@router.post("/markdown-to-docx")
async def markdown_to_docx(
    data: MarkdownToDocxRequest,
    user: UserData = Depends(auth),
) -> StreamingResponse:
    """Convert a Markdown string to a real, RTL-correct .docx file."""
    ast = parse_markdown(data.markdown, title=data.title)
    buf = render_docx(ast)
    return StreamingResponse(
        buf,
        media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        headers={"Content-Disposition": 'attachment; filename="document.docx"'},
    )
