"""Fixtures for file uploads and processing."""

import base64
import struct
from io import BytesIO

import pytest


@pytest.fixture
def mock_png_bytes() -> bytes:
    """Create minimal valid PNG bytes (1x1 red pixel)."""
    # Minimal 1x1 red PNG
    png_data = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR"  # IHDR chunk length + type
        b"\x00\x00\x00\x01"  # width = 1
        b"\x00\x00\x00\x01"  # height = 1
        b"\x08\x02"  # bit depth=8, color type=2 (RGB)
        b"\x00\x00\x00"  # compression, filter, interlace
        b"\x90wS\xde"  # CRC
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"  # IDAT
        b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND
    )
    return png_data


@pytest.fixture
def mock_image_file(mock_png_bytes: bytes) -> BytesIO:
    """Create a mock PNG image file as BytesIO."""
    buf = BytesIO(mock_png_bytes)
    buf.seek(0)
    return buf


@pytest.fixture
def mock_pdf_bytes() -> bytes:
    """Create minimal valid PDF bytes."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f\n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n9\n%%EOF"
    )


@pytest.fixture
def mock_pdf_content(mock_pdf_bytes: bytes) -> BytesIO:
    """Create a mock PDF file as BytesIO."""
    buf = BytesIO(mock_pdf_bytes)
    buf.seek(0)
    return buf


@pytest.fixture
def mock_audio_bytes() -> bytes:
    """Create minimal valid WAV audio bytes."""
    # Minimal WAV file: 44-byte header + 1 second of silence at 8000 Hz
    sample_rate = 8000
    num_samples = sample_rate  # 1 second
    data_size = num_samples * 2  # 16-bit samples
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM chunk size
        1,  # PCM format
        1,  # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,  # block align
        16,  # bits per sample
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size


@pytest.fixture
def mock_audio_file(mock_audio_bytes: bytes) -> BytesIO:
    """Create a mock WAV audio file as BytesIO."""
    buf = BytesIO(mock_audio_bytes)
    buf.seek(0)
    return buf


@pytest.fixture
def base64_png(mock_png_bytes: bytes) -> str:
    """Create a base64-encoded PNG data URL."""
    encoded = base64.b64encode(mock_png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@pytest.fixture
def base64_pdf(mock_pdf_bytes: bytes) -> str:
    """Create a base64-encoded PDF data URL."""
    encoded = base64.b64encode(mock_pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


@pytest.fixture
def base64_audio(mock_audio_bytes: bytes) -> str:
    """Create a base64-encoded WAV audio data URL."""
    encoded = base64.b64encode(mock_audio_bytes).decode("utf-8")
    return f"data:audio/wav;base64,{encoded}"
