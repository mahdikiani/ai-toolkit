"""OCR services for text extraction using OpenRouter API via httpx."""

import asyncio
import base64
import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image

from server.config import Settings
from utils.files import imagetools
from utils.integrations.openrouter import complete_chat_json


@lru_cache(maxsize=1)
def _read_ocr_prompt() -> str:
    """Read OCR prompt from file."""
    with open(Path(__file__).resolve().parent / "ocr.prompt") as f:
        return f.read()


@lru_cache(maxsize=1)
def _read_text_enhancement_prompt() -> str:
    """Read OCR prompt from file."""
    with open(Path(__file__).resolve().parent / "text_enhancement.prompt") as f:
        return f.read()


async def text_enhancement(
    text: str,
    model: str = "google/gemini-2.5-flash",
) -> str:
    """Enhance OCR text quality using LLM post-processing."""
    prompt = _read_text_enhancement_prompt()
    response = await complete_chat_json({
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    })
    return response["choices"][0]["message"]["content"].strip()


MAX_DIM = 2800
FALLBACK_DIM = 1600
FALLBACK_QUALITY = 60


async def ocr_to_text(
    image: BytesIO,
    model: str | None = None,
    layout_hint: str = "",
) -> str:
    """Extract text from a page image with smart fallback.

    - First attempt: 2800px max-dim, JPEG quality 85.
    - If the API *errors* (timeout, 413, 5xx), retry once at 1600px / quality 60.
    - If the API returns empty content, accept it — the page may legitimately
      contain only images/charts (the prompt inserts ![description](#) for those).
    """
    prompt = _read_ocr_prompt()
    image.seek(0)

    with Image.open(image) as opened_image:
        img = opened_image.copy()

    result, api_error = await _ocr_attempt(
        img, prompt, model, layout_hint, max_dim=MAX_DIM, quality=85
    )
    if result is not None:
        return result

    if api_error:
        logger.warning("OCR API error; retrying with fallback resolution")
        result, _ = await _ocr_attempt(
            img, prompt, model, layout_hint,
            max_dim=FALLBACK_DIM, quality=FALLBACK_QUALITY,
        )
        return result or ""

    return ""


async def _ocr_attempt(
    img: Image.Image,
    prompt: str,
    model: str | None,
    layout_hint: str,
    *,
    max_dim: int,
    quality: int,
) -> tuple[str | None, bool]:
    """Try one OCR call.

    Returns (text | None, had_api_error).
    - None + False → API returned empty (OK, page might be blank/image-only)
    - None + True  → API error (caller may retry with lower res)
    """
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_data}"

    user_content: list[dict] = [
        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
    ]
    if layout_hint:
        user_content.insert(0, {"type": "text", "text": layout_hint})

    try:
        response = await complete_chat_json({
            "model": model or Settings.ocr_vlm_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
        })
        text = response["choices"][0]["message"]["content"].strip()
        if text:
            logger.info(
                "OCR %d chars from %dx%d image (max_dim=%d, quality=%d)",
                len(text), img.width, img.height, max_dim, quality,
            )
            return text, False
        logger.info(
            "OCR empty (page %dx%d may be blank/image-only; max_dim=%d)",
            img.width, img.height, max_dim,
        )
        return None, False
    except Exception as exc:
        logger.warning(
            "OCR API error for %dx%d (max_dim=%d, quality=%d): %s",
            img.width, img.height, max_dim, quality, exc,
        )
        return None, True


async def process_pages_batch(
    pages: list[BytesIO], max_concurrent: int = 10
) -> list[str]:
    """Process multiple pages concurrently with semaphore control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_page(page: BytesIO) -> str:
        async with semaphore:
            text = await ocr_to_text(page)
            return await text_enhancement(text)

    return await asyncio.gather(*(process_single_page(page) for page in pages))


# Image preprocessing functions following Single Responsibility Principle
def _prepare_pdf_pages(file_content: BytesIO) -> list[BytesIO]:
    """Extract and convert PDF pages to JPEG bytes."""
    from utils.files import pdftools

    pages_image: list[Image.Image] = pdftools.extract_pdf_bytes_pages(file_content)
    return [imagetools.convert_to_jpg_bytes(page) for page in pages_image]


def _prepare_converting_image(file_content: BytesIO) -> list[BytesIO]:
    """Convert non-JPEG images to JPEG bytes."""
    return [imagetools.convert_to_jpg_bytes(file_content)]


def _prepare_jpeg_image(file_content: BytesIO) -> list[BytesIO]:
    """Prepare JPEG image for processing."""
    return [file_content]


def prepare_pages(file_content: BytesIO, file_type: str) -> list[BytesIO]:
    """Prepare pages for OCR processing based on file type."""
    if file_type == "application/pdf":
        return _prepare_pdf_pages(file_content)
    elif file_type in {"image/png", "image/tiff", "image/webp"}:
        return _prepare_converting_image(file_content)
    elif file_type == "image/jpeg":
        return _prepare_jpeg_image(file_content)
    else:
        return []
