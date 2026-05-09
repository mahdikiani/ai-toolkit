"""PaddleOCR services for local OCR processing."""

import asyncio
import logging
import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from server.config import Settings


@lru_cache(maxsize=1)
def _get_pipeline() -> object:
    """Get or create the PaddleOCR pipeline with caching."""
    import paddlex as pdx

    return pdx.create_pipeline(
        pipeline=Settings.paddle_pipeline_name,
        device=Settings.paddle_device,
    )


def _extract_text_payload(payload: object) -> str:
    """Extract text content from PaddleOCR result payload."""
    if payload is None:
        return ""
    if isinstance(payload, dict):
        for key in ("markdown", "text", "rec_text", "ocr_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value

        chunks: list[str] = []
        parsing_items = payload.get("parsing_res_list")
        if isinstance(parsing_items, list):
            for item in parsing_items:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text)
        if chunks:
            return "\n".join(chunks)

    raw = str(payload).strip()
    return raw


def _predict_single_image(page: BytesIO) -> str | None:
    """Run OCR prediction on a single image page."""
    pipeline = _get_pipeline()
    page.seek(0)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        tmp_file.write(page.read())
        tmp_path = Path(tmp_file.name)

    try:
        result_iter = pipeline.predict(str(tmp_path))
        result = next(iter(result_iter), None)
        if result is None:
            return None

        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        text = _extract_text_payload(payload)
        return text or None
    finally:
        tmp_path.unlink(missing_ok=True)


async def process_pages_with_paddle(pages: list[BytesIO]) -> list[str | None]:
    """Process multiple pages concurrently using PaddleOCR."""

    async def _run(page: BytesIO) -> str | None:
        try:
            return await asyncio.to_thread(_predict_single_image, page)
        except Exception:
            logging.exception("Paddle OCR extraction failed")
            return None

    return await asyncio.gather(*(_run(page) for page in pages))
