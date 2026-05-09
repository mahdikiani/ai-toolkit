"""
Shared utilities for language services.

This module provides common file handling utilities used across all language
services (chat, prompts, executions, translate) for consistent file validation
and content type detection.
"""

from pathlib import Path

from apps.language.shared.schemas import ContentType


class FileValidationError(Exception):
    """Raised when file validation fails."""

    pass


def validate_file_url(file_url: str) -> bool:
    """
    Validate that a file URL is well-formed.

    Args:
        file_url: URL or path to validate

    Returns:
        True if valid

    Raises:
        FileValidationError: If URL is invalid
    """
    if not file_url or not file_url.strip():
        raise FileValidationError("File URL cannot be empty")

    # Add more validation as needed (S3 URLs, local paths, etc.)
    return True


def detect_content_type(file_url: str) -> ContentType:
    """
    Determine content type from file extension.

    Args:
        file_url: File URL or path

    Returns:
        ContentType enum value

    Raises:
        FileValidationError: If file type is unsupported
    """
    # Remove query parameters and fragments from URL
    # Split on '?' and '#' to get just the path
    clean_url = file_url.split("?")[0].split("#")[0]

    ext = Path(clean_url).suffix.lower()

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    document_extensions = {".pdf", ".doc", ".docx", ".txt", ".md"}

    if ext in image_extensions:
        return ContentType.IMAGE
    elif ext in document_extensions:
        return ContentType.DOCUMENT
    else:
        raise FileValidationError(f"Unsupported file type: {ext}")


def validate_file_size(file_url: str, max_size_mb: int = 10) -> bool:
    """
    Validate file size is within limits.

    Args:
        file_url: File URL or path
        max_size_mb: Maximum file size in megabytes

    Returns:
        True if valid

    Raises:
        FileValidationError: If file exceeds size limit
    """
    # Implementation depends on storage backend (S3, local, etc.)
    # For now, return True as placeholder
    return True
