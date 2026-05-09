"""OCR services for text extraction using OpenRouter API."""

import asyncio
import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

from server.config import Settings
from utils import b64tools, imagetools

# Global OCR client for reuse
_ocr_client = None


def _get_ocr_client() -> AsyncOpenAI:
    """Get or create OCR client following lazy initialization pattern."""
    global _ocr_client
    if _ocr_client is None:
        _ocr_client = AsyncOpenAI(
            base_url=Settings.openrouter_base_url, api_key=Settings.openrouter_api_key
        )
    return _ocr_client


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
    client = _get_ocr_client()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content.strip()


async def ocr_to_text(
    image: BytesIO,
    model: str = "google/gemini-2.5-flash-lite",
) -> str | None:
    """Extract text from an image using OpenRouter API."""
    try:
        client = _get_ocr_client()
        prompt = _read_ocr_prompt()
        data_url = b64tools.b64_file(image)

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": [{"type": "text", "text": prompt}]},
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": data_url}}],
                },
            ],
        )
    except Exception:
        # Log error but don't raise to allow other pages to process
        logging.exception("OCR extraction failed")
        return None

    return response.choices[0].message.content.strip()


async def process_pages_batch(
    pages: list[BytesIO], max_concurrent: int = 10
) -> list[str | None]:
    """Process multiple pages concurrently with semaphore control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single_page(page: BytesIO) -> str | None:
        async with semaphore:
            text = await ocr_to_text(page)
            if text:
                return await text_enhancement(text)
            return None

    return await asyncio.gather(*(process_single_page(page) for page in pages))


# Image preprocessing functions following Single Responsibility Principle
def _prepare_pdf_pages(file_content: BytesIO) -> list[BytesIO]:
    """Extract and convert PDF pages to JPEG bytes."""
    from utils import pdftools

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
