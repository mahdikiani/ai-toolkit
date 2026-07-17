"""Utilities for converting files and images to base64 data URIs."""

import base64
import mimetypes
from io import BytesIO
from pathlib import Path

import magic
from PIL import Image


def _b64_image(image: Image.Image) -> str:
    with BytesIO() as output:
        image.save(output, format="jpeg")
        output.seek(0)
        mime_type = "image/jpeg"
        b64_data = base64.b64encode(output.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"


def _b64_file(file_path: Path | str) -> str:
    mime_type = mimetypes.guess_type(file_path)[0]
    with open(file_path, "rb") as file:
        b64_data = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64_data}"


def _b64_bytes(data: BytesIO) -> str:
    # INSERT_YOUR_CODE
    # Read the first 2048 bytes to detect the mimetype from the file signature (header)
    current_pos = data.tell()
    data.seek(0)
    header = data.read(2048)
    data.seek(current_pos)

    mime_type = magic.from_buffer(header, mime=True)
    data.seek(0)
    b64_data = base64.b64encode(data.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64_data}"


def b64_file(file_path: Path | str | Image.Image) -> str:
    """
    Convert a file, image, or BytesIO to a base64 data URI.

    Args:
        file_path: Path to file, PIL Image, or BytesIO object.

    Returns:
        Base64-encoded data URI string with appropriate MIME type.
    """
    if isinstance(file_path, Image.Image):
        return _b64_image(file_path)
    if isinstance(file_path, BytesIO):
        return _b64_bytes(file_path)
    return _b64_file(file_path)
