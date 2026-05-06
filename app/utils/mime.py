"""Provide module functionality."""
from io import BytesIO
from pathlib import Path

import magic


def check_path_file_type(file_path: Path) -> str:
    """Check and validate file MIME type."""
    with open(file_path, "rb") as file:
        return check_file_type(BytesIO(file.read(2048)))


def check_file_type(file: BytesIO) -> str:
    """
    Check and validate file MIME type.

    Args:
        file: BytesIO object containing file data
        accepted_mimes: List of accepted MIME types

    Returns:
        str: Detected MIME type

    Raises:
        BaseHTTPException: If file type is not supported
    """

    file.seek(0)  # Reset the file pointer to the beginning

    # Initialize the magic MIME type detector
    mime_detector = magic.Magic(mime=True)

    # Detect MIME type from the buffer
    file_data = file.read(8192)  # Read more bytes for better detection
    mime_type = mime_detector.from_buffer(file_data)

    file.seek(0)  # Reset the file pointer to the beginning

    # Fallback for ZIP files that might be detected as octet-stream
    if mime_type == "application/octet-stream":
        file_data = file.read(1024)
        if file_data.startswith(b"PK\x03\x04") or file_data.startswith(b"PK\x05\x06"):
            mime_type = "application/zip"
        file.seek(0)

    return mime_type
