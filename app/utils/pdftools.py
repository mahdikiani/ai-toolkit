"""PDF processing utilities for extracting pages as images."""

from io import BytesIO
from pathlib import Path

import pdf2image
from PIL import Image


def number_of_pages(path: Path) -> int:
    """Get number of pages in PDF file using pdf2image for efficiency."""
    info = pdf2image.pdfinfo_from_path(str(path))
    return int(info.get("Pages", 0))


def extract_pdf_pages(path: Path) -> list[Image.Image]:
    """
    Extract all pages from a PDF file as PIL Images.

    Args:
        path: Path to the PDF file.

    Returns:
        List of PIL Image objects, one per page.
    """
    return pdf2image.convert_from_path(str(path))


def extract_pdf_pages_with_index(path: Path) -> list[tuple[int, Image.Image]]:
    """
    Extract all pages from a PDF file with their page indices.

    Args:
        path: Path to the PDF file.

    Returns:
        List of tuples containing (page_index, PIL Image).
    """
    return list(enumerate(pdf2image.convert_from_path(str(path))))


def extract_pdf_bytes_pages(pdf_bytes: BytesIO) -> list[Image.Image]:
    """
    Extract all pages from PDF bytes as PIL Images.

    Args:
        pdf_bytes: BytesIO object containing PDF data.

    Returns:
        List of PIL Image objects, one per page.
    """
    pdf_bytes.seek(0)
    return pdf2image.convert_from_bytes(pdf_bytes.read())


def extract_pdf_bytes_pages_with_index(
    pdf_bytes: BytesIO,
) -> list[tuple[int, Image.Image]]:
    """
    Extract all pages from PDF bytes with their page indices.

    Args:
        pdf_bytes: BytesIO object containing PDF data.

    Returns:
        List of tuples containing (page_index, PIL Image).
    """
    pdf_bytes.seek(0)
    return list(enumerate(pdf2image.convert_from_bytes(pdf_bytes.read())))
