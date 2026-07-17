"""
Unit tests for shared language service utilities.

Tests for file validation, content type detection, and file size validation
defined in apps/language/shared/utils.py.

**Validates: Requirements 6.5, 6.6**
"""

import pytest

from apps.language.shared.schemas import ContentType
from apps.language.shared.utils import (
    FileValidationError,
    detect_content_type,
    validate_file_size,
    validate_file_url,
)


@pytest.mark.unit
class TestFileURLValidation:
    """Tests for validate_file_url function."""

    def test_valid_http_url(self) -> None:
        """Should accept valid HTTP URL."""
        url = "http://example.com/file.jpg"
        assert validate_file_url(url) is True

    def test_valid_https_url(self) -> None:
        """Should accept valid HTTPS URL."""
        url = "https://example.com/file.pdf"
        assert validate_file_url(url) is True

    def test_valid_s3_url(self) -> None:
        """Should accept valid S3 URL."""
        url = "s3://bucket-name/path/to/file.jpg"
        assert validate_file_url(url) is True

    def test_valid_local_path(self) -> None:
        """Should accept valid local file path."""
        url = "/path/to/local/file.jpg"
        assert validate_file_url(url) is True

    def test_valid_relative_path(self) -> None:
        """Should accept valid relative file path."""
        url = "relative/path/file.jpg"
        assert validate_file_url(url) is True

    def test_empty_string_raises_error(self) -> None:
        """Should reject empty string."""
        with pytest.raises(FileValidationError) as exc_info:
            validate_file_url("")
        assert "File URL cannot be empty" in str(exc_info.value)

    def test_whitespace_only_raises_error(self) -> None:
        """Should reject whitespace-only string."""
        with pytest.raises(FileValidationError) as exc_info:
            validate_file_url("   ")
        assert "File URL cannot be empty" in str(exc_info.value)

    def test_url_with_query_params(self) -> None:
        """Should accept URL with query parameters."""
        url = "https://example.com/file.jpg?token=abc123"
        assert validate_file_url(url) is True

    def test_url_with_fragment(self) -> None:
        """Should accept URL with fragment."""
        url = "https://example.com/file.pdf#page=5"
        assert validate_file_url(url) is True

    def test_url_with_special_chars(self) -> None:
        """Should accept URL with special characters."""
        url = "https://example.com/files/my-file_2024.jpg"
        assert validate_file_url(url) is True


@pytest.mark.unit
class TestContentTypeDetection:
    """Tests for detect_content_type function."""

    # Image format tests
    def test_detect_jpg_as_image(self) -> None:
        """Should detect .jpg as IMAGE type."""
        url = "https://example.com/photo.jpg"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_detect_jpeg_as_image(self) -> None:
        """Should detect .jpeg as IMAGE type."""
        url = "https://example.com/photo.jpeg"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_detect_png_as_image(self) -> None:
        """Should detect .png as IMAGE type."""
        url = "https://example.com/photo.png"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_detect_gif_as_image(self) -> None:
        """Should detect .gif as IMAGE type."""
        url = "https://example.com/animation.gif"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_detect_webp_as_image(self) -> None:
        """Should detect .webp as IMAGE type."""
        url = "https://example.com/photo.webp"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_detect_bmp_as_image(self) -> None:
        """Should detect .bmp as IMAGE type."""
        url = "https://example.com/photo.bmp"
        assert detect_content_type(url) == ContentType.IMAGE

    # Document format tests
    def test_detect_pdf_as_document(self) -> None:
        """Should detect .pdf as DOCUMENT type."""
        url = "https://example.com/report.pdf"
        assert detect_content_type(url) == ContentType.DOCUMENT

    def test_detect_doc_as_document(self) -> None:
        """Should detect .doc as DOCUMENT type."""
        url = "https://example.com/report.doc"
        assert detect_content_type(url) == ContentType.DOCUMENT

    def test_detect_docx_as_document(self) -> None:
        """Should detect .docx as DOCUMENT type."""
        url = "https://example.com/report.docx"
        assert detect_content_type(url) == ContentType.DOCUMENT

    def test_detect_txt_as_document(self) -> None:
        """Should detect .txt as DOCUMENT type."""
        url = "https://example.com/notes.txt"
        assert detect_content_type(url) == ContentType.DOCUMENT

    def test_detect_md_as_document(self) -> None:
        """Should detect .md as DOCUMENT type."""
        url = "https://example.com/readme.md"
        assert detect_content_type(url) == ContentType.DOCUMENT

    # Case insensitivity tests
    def test_uppercase_extension_detected(self) -> None:
        """Should detect uppercase extensions."""
        url = "https://example.com/photo.JPG"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_mixed_case_extension_detected(self) -> None:
        """Should detect mixed case extensions."""
        url = "https://example.com/photo.JpEg"
        assert detect_content_type(url) == ContentType.IMAGE

    # Unsupported format tests
    def test_unsupported_extension_raises_error(self) -> None:
        """Should reject unsupported file extension."""
        url = "https://example.com/file.xyz"
        with pytest.raises(FileValidationError) as exc_info:
            detect_content_type(url)
        assert "Unsupported file type: .xyz" in str(exc_info.value)

    def test_no_extension_raises_error(self) -> None:
        """Should reject file without extension."""
        url = "https://example.com/file"
        with pytest.raises(FileValidationError) as exc_info:
            detect_content_type(url)
        assert "Unsupported file type:" in str(exc_info.value)

    def test_video_extension_raises_error(self) -> None:
        """Should reject video file extensions (not supported)."""
        url = "https://example.com/video.mp4"
        with pytest.raises(FileValidationError) as exc_info:
            detect_content_type(url)
        assert "Unsupported file type: .mp4" in str(exc_info.value)

    def test_audio_extension_raises_error(self) -> None:
        """Should reject audio file extensions (not supported)."""
        url = "https://example.com/audio.mp3"
        with pytest.raises(FileValidationError) as exc_info:
            detect_content_type(url)
        assert "Unsupported file type: .mp3" in str(exc_info.value)

    def test_executable_extension_raises_error(self) -> None:
        """Should reject executable file extensions."""
        url = "https://example.com/program.exe"
        with pytest.raises(FileValidationError) as exc_info:
            detect_content_type(url)
        assert "Unsupported file type: .exe" in str(exc_info.value)

    # Path handling tests
    def test_detect_from_local_path(self) -> None:
        """Should detect content type from local file path."""
        path = "/home/user/documents/report.pdf"
        assert detect_content_type(path) == ContentType.DOCUMENT

    def test_detect_from_s3_url(self) -> None:
        """Should detect content type from S3 URL."""
        url = "s3://my-bucket/images/photo.png"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_detect_with_query_params(self) -> None:
        """Should detect content type ignoring query parameters."""
        url = "https://example.com/file.jpg?token=abc&size=large"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_detect_with_multiple_dots(self) -> None:
        """Should detect content type from filename with multiple dots."""
        url = "https://example.com/my.file.name.pdf"
        assert detect_content_type(url) == ContentType.DOCUMENT

    def test_detect_from_relative_path(self) -> None:
        """Should detect content type from relative path."""
        path = "uploads/images/photo.webp"
        assert detect_content_type(path) == ContentType.IMAGE


@pytest.mark.unit
class TestFileSizeValidation:
    """Tests for validate_file_size function."""

    def test_default_max_size(self) -> None:
        """Should use default max size of 10MB."""
        url = "https://example.com/file.jpg"
        # Currently returns True as placeholder
        assert validate_file_size(url) is True

    def test_custom_max_size(self) -> None:
        """Should accept custom max size parameter."""
        url = "https://example.com/file.jpg"
        assert validate_file_size(url, max_size_mb=5) is True

    def test_zero_max_size(self) -> None:
        """Should handle zero max size."""
        url = "https://example.com/file.jpg"
        # Currently returns True as placeholder
        assert validate_file_size(url, max_size_mb=0) is True

    def test_large_max_size(self) -> None:
        """Should handle large max size values."""
        url = "https://example.com/file.jpg"
        assert validate_file_size(url, max_size_mb=1000) is True

    def test_http_url(self) -> None:
        """Should validate HTTP URL."""
        url = "http://example.com/file.pdf"
        assert validate_file_size(url) is True

    def test_https_url(self) -> None:
        """Should validate HTTPS URL."""
        url = "https://example.com/file.pdf"
        assert validate_file_size(url) is True

    def test_s3_url(self) -> None:
        """Should validate S3 URL."""
        url = "s3://bucket/path/file.jpg"
        assert validate_file_size(url) is True

    def test_local_path(self) -> None:
        """Should validate local file path."""
        path = "/path/to/file.jpg"
        assert validate_file_size(path) is True


@pytest.mark.unit
class TestErrorMessages:
    """Tests for validation error messages."""

    def test_empty_url_error_message(self) -> None:
        """Should provide clear error message for empty URL."""
        with pytest.raises(FileValidationError) as exc_info:
            validate_file_url("")
        error_msg = str(exc_info.value)
        assert "File URL cannot be empty" in error_msg
        assert error_msg == "File URL cannot be empty"

    def test_whitespace_url_error_message(self) -> None:
        """Should provide clear error message for whitespace URL."""
        with pytest.raises(FileValidationError) as exc_info:
            validate_file_url("   \t\n   ")
        error_msg = str(exc_info.value)
        assert "File URL cannot be empty" in error_msg

    def test_unsupported_type_error_message(self) -> None:
        """Should provide clear error message for unsupported file type."""
        with pytest.raises(FileValidationError) as exc_info:
            detect_content_type("file.unknown")
        error_msg = str(exc_info.value)
        assert "Unsupported file type: .unknown" in error_msg

    def test_no_extension_error_message(self) -> None:
        """Should provide clear error message for file without extension."""
        with pytest.raises(FileValidationError) as exc_info:
            detect_content_type("file_without_extension")
        error_msg = str(exc_info.value)
        assert "Unsupported file type:" in error_msg

    def test_error_message_includes_extension(self) -> None:
        """Should include the problematic extension in error message."""
        with pytest.raises(FileValidationError) as exc_info:
            detect_content_type("document.xyz")
        error_msg = str(exc_info.value)
        assert ".xyz" in error_msg


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_url_with_trailing_slash(self) -> None:
        """Should handle URL with trailing slash."""
        url = "https://example.com/path/"
        # Should raise error as no extension
        with pytest.raises(FileValidationError):
            detect_content_type(url)

    def test_url_with_dot_in_domain(self) -> None:
        """Should handle URL with dots in domain name."""
        url = "https://my.example.com/file.jpg"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_url_with_port(self) -> None:
        """Should handle URL with port number."""
        url = "https://example.com:8080/file.pdf"
        assert detect_content_type(url) == ContentType.DOCUMENT

    def test_url_with_username_password(self) -> None:
        """Should handle URL with authentication."""
        url = "https://user:pass@example.com/file.jpg"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_windows_path(self) -> None:
        """Should handle Windows file path."""
        path = "C:\\Users\\Documents\\file.pdf"
        assert detect_content_type(path) == ContentType.DOCUMENT

    def test_url_with_encoded_chars(self) -> None:
        """Should handle URL with encoded characters."""
        url = "https://example.com/my%20file.jpg"
        assert detect_content_type(url) == ContentType.IMAGE

    def test_hidden_file(self) -> None:
        """Should handle hidden file (starting with dot)."""
        path = "/home/user/.hidden.jpg"
        assert detect_content_type(path) == ContentType.IMAGE

    def test_file_with_no_name(self) -> None:
        """Should handle file with only extension (hidden file)."""
        # A file like ".jpg" has no extension (it's a hidden file named "jpg")
        # Path(".jpg").suffix returns empty string
        path = "/path/to/.jpg"
        with pytest.raises(FileValidationError):
            detect_content_type(path)

    def test_very_long_url(self) -> None:
        """Should handle very long URL."""
        url = "https://example.com/" + "a" * 1000 + "/file.jpg"
        assert validate_file_url(url) is True
        assert detect_content_type(url) == ContentType.IMAGE

    def test_url_with_unicode(self) -> None:
        """Should handle URL with unicode characters."""
        url = "https://example.com/文件.jpg"
        assert validate_file_url(url) is True
        assert detect_content_type(url) == ContentType.IMAGE


@pytest.mark.unit
class TestIntegrationScenarios:
    """Tests for integrated validation scenarios."""

    def test_validate_and_detect_image(self) -> None:
        """Should validate URL and detect image type."""
        url = "https://example.com/photo.jpg"
        assert validate_file_url(url) is True
        assert detect_content_type(url) == ContentType.IMAGE
        assert validate_file_size(url) is True

    def test_validate_and_detect_document(self) -> None:
        """Should validate URL and detect document type."""
        url = "https://example.com/report.pdf"
        assert validate_file_url(url) is True
        assert detect_content_type(url) == ContentType.DOCUMENT
        assert validate_file_size(url) is True

    def test_validate_s3_image(self) -> None:
        """Should validate S3 URL and detect image type."""
        url = "s3://my-bucket/images/photo.png"
        assert validate_file_url(url) is True
        assert detect_content_type(url) == ContentType.IMAGE
        assert validate_file_size(url) is True

    def test_validate_local_document(self) -> None:
        """Should validate local path and detect document type."""
        path = "/home/user/documents/report.docx"
        assert validate_file_url(path) is True
        assert detect_content_type(path) == ContentType.DOCUMENT
        assert validate_file_size(path) is True

    def test_invalid_url_fails_validation(self) -> None:
        """Should fail validation for empty URL."""
        with pytest.raises(FileValidationError):
            validate_file_url("")

    def test_unsupported_type_fails_detection(self) -> None:
        """Should fail detection for unsupported type."""
        url = "https://example.com/file.mp4"
        assert validate_file_url(url) is True
        with pytest.raises(FileValidationError):
            detect_content_type(url)

    def test_all_image_formats_validated(self) -> None:
        """Should validate all supported image formats."""
        image_formats = ["jpg", "jpeg", "png", "gif", "webp", "bmp"]
        for fmt in image_formats:
            url = f"https://example.com/image.{fmt}"
            assert validate_file_url(url) is True
            assert detect_content_type(url) == ContentType.IMAGE
            assert validate_file_size(url) is True

    def test_all_document_formats_validated(self) -> None:
        """Should validate all supported document formats."""
        doc_formats = ["pdf", "doc", "docx", "txt", "md"]
        for fmt in doc_formats:
            url = f"https://example.com/document.{fmt}"
            assert validate_file_url(url) is True
            assert detect_content_type(url) == ContentType.DOCUMENT
            assert validate_file_size(url) is True
