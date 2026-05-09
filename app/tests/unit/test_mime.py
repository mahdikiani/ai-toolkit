"""Unit tests for MIME type detection utilities."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from utils.mime import check_file_type


@pytest.mark.unit
class TestCheckFileType:
    """Tests for check_file_type function."""

    def test_detects_pdf(self, mock_pdf_bytes: bytes) -> None:
        """check_file_type should detect PDF files."""
        with patch("utils.mime.magic.Magic") as mock_magic_cls:
            mock_magic = MagicMock()
            mock_magic.from_buffer.return_value = "application/pdf"
            mock_magic_cls.return_value = mock_magic

            result = check_file_type(BytesIO(mock_pdf_bytes))

        assert result == "application/pdf"

    def test_detects_jpeg(self) -> None:
        """check_file_type should detect JPEG images."""
        # JPEG magic bytes
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        with patch("utils.mime.magic.Magic") as mock_magic_cls:
            mock_magic = MagicMock()
            mock_magic.from_buffer.return_value = "image/jpeg"
            mock_magic_cls.return_value = mock_magic

            result = check_file_type(BytesIO(jpeg_bytes))

        assert result == "image/jpeg"

    def test_detects_png(self, mock_png_bytes: bytes) -> None:
        """check_file_type should detect PNG images."""
        with patch("utils.mime.magic.Magic") as mock_magic_cls:
            mock_magic = MagicMock()
            mock_magic.from_buffer.return_value = "image/png"
            mock_magic_cls.return_value = mock_magic

            result = check_file_type(BytesIO(mock_png_bytes))

        assert result == "image/png"

    def test_detects_zip_from_magic_bytes(self) -> None:
        """check_file_type should detect ZIP from magic bytes when octet-stream."""
        zip_bytes = b"PK\x03\x04" + b"\x00" * 100

        with patch("utils.mime.magic.Magic") as mock_magic_cls:
            mock_magic = MagicMock()
            # First call returns octet-stream, triggering ZIP fallback
            mock_magic.from_buffer.return_value = "application/octet-stream"
            mock_magic_cls.return_value = mock_magic

            result = check_file_type(BytesIO(zip_bytes))

        assert result == "application/zip"

    def test_resets_file_pointer(self) -> None:
        """check_file_type should reset file pointer to beginning after reading."""
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        buf = BytesIO(data)
        buf.seek(50)  # Move pointer to middle

        with patch("utils.mime.magic.Magic") as mock_magic_cls:
            mock_magic = MagicMock()
            mock_magic.from_buffer.return_value = "image/jpeg"
            mock_magic_cls.return_value = mock_magic

            check_file_type(buf)

        # File pointer should be reset to 0
        assert buf.tell() == 0
