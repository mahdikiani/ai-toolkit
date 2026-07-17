"""Unit tests for OCR schemas."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


@pytest.mark.unit
class TestOcrTaskSchemaCreate:
    """Tests for OcrTaskSchemaCreate schema."""

    def test_is_pdf_returns_true_for_pdf_url(self) -> None:
        """is_pdf should return True for .pdf URLs."""
        from apps.ocr.schemas import OcrTaskSchemaCreate

        schema = OcrTaskSchemaCreate(file_url="https://example.com/doc.pdf")
        assert schema.is_pdf is True

    def test_is_pdf_returns_false_for_non_pdf(self) -> None:
        """is_pdf should return False for non-PDF URLs."""
        from apps.ocr.schemas import OcrTaskSchemaCreate

        schema = OcrTaskSchemaCreate(file_url="https://example.com/img.png")
        assert schema.is_pdf is False

    async def test_file_content_from_base64(self, mock_png_bytes: bytes) -> None:
        """file_content should decode base64 data URLs."""
        from apps.ocr.schemas import OcrTaskSchemaCreate

        encoded = base64.b64encode(mock_png_bytes).decode("utf-8")
        schema = OcrTaskSchemaCreate(file_url=f"data:image/png;base64,{encoded}")

        content = await schema.file_content()
        assert content.read() == mock_png_bytes

    async def test_file_content_handles_invalid_base64(self) -> None:
        """file_content should not crash on invalid base64 data."""
        from apps.ocr.schemas import OcrTaskSchemaCreate

        schema = OcrTaskSchemaCreate(file_url="data:image/png;base64,!!!invalid!!!")

        content = await schema.file_content()
        assert content.read() == b""

    async def test_file_content_from_url(self) -> None:
        """file_content should fetch content from HTTP URLs."""
        from apps.ocr.schemas import OcrTaskSchemaCreate

        schema = OcrTaskSchemaCreate(file_url="https://example.com/file.pdf")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_response = MagicMock()
            mock_response.content = b"pdf content"
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            content = await schema.file_content()
            assert content.read() == b"pdf content"

    async def test_file_content_base64(self, mock_png_bytes: bytes) -> None:
        """file_content_base64 should return base64-encoded content."""
        from apps.ocr.schemas import OcrTaskSchemaCreate

        encoded = base64.b64encode(mock_png_bytes).decode("utf-8")
        schema = OcrTaskSchemaCreate(file_url=f"data:image/png;base64,{encoded}")

        result = await schema.file_content_base64()
        assert result == encoded


@pytest.mark.unit
class TestOcrTaskUploadFormSchema:
    """Tests for OcrTaskUploadFormSchema schema."""

    def test_as_form_parses_json_fields(self) -> None:
        """as_form should parse JSON string fields into dicts."""
        from apps.ocr.schemas import OcrTaskUploadFormSchema

        result = OcrTaskUploadFormSchema.as_form(
            user_id="user_123",
            webhook_url="https://hook.example.com",
            webhook_custom_headers='{"Authorization": "Bearer test"}',
            meta_data='{"source": "test"}',
            ocr_engine="llm",
        )

        assert result.user_id == "user_123"
        assert result.webhook_url == "https://hook.example.com"
        assert result.webhook_custom_headers == {"Authorization": "Bearer test"}
        assert result.meta_data == {"source": "test"}
        assert result.ocr_engine == "llm"

    def test_as_form_handles_none_json_fields(self) -> None:
        """as_form should handle None JSON fields."""
        from apps.ocr.schemas import OcrTaskUploadFormSchema

        result = OcrTaskUploadFormSchema.as_form(
            user_id=None,
            webhook_url=None,
            webhook_custom_headers=None,
            meta_data=None,
            ocr_engine=None,
        )

        assert result.webhook_custom_headers is None
        assert result.meta_data is None

    def test_as_form_raises_on_invalid_json(self) -> None:
        """as_form should raise HTTPException on invalid JSON."""
        from apps.ocr.schemas import OcrTaskUploadFormSchema

        with pytest.raises(HTTPException) as exc:
            OcrTaskUploadFormSchema.as_form(
                user_id=None,
                webhook_url=None,
                webhook_custom_headers="{invalid json}",
                meta_data=None,
                ocr_engine=None,
            )

        assert isinstance(exc.value, HTTPException)
        assert exc.value.status_code == 422


@pytest.mark.unit
class TestOcrTaskBase64Schema:
    """Tests for OcrTaskBase64Schema schema."""

    def test_to_create_schema_builds_data_url(self) -> None:
        """to_create_schema should build data URL from base64 content."""
        from apps.ocr.schemas import OcrEngineType, OcrTaskBase64Schema

        schema = OcrTaskBase64Schema(
            content_base64="dGVzdA==",
            mime_type="image/png",
            user_id="user_123",
            ocr_engine=OcrEngineType.llm,
        )

        create = schema.to_create_schema()
        assert create.file_url == "data:image/png;base64,dGVzdA=="
        assert create.user_id == "user_123"
        assert create.ocr_engine == "llm"

    def test_to_create_schema_preserves_data_url(self) -> None:
        """to_create_schema should not wrap if already a data URL."""
        from apps.ocr.schemas import OcrTaskBase64Schema

        schema = OcrTaskBase64Schema(
            content_base64="data:image/png;base64,dGVzdA==",
        )

        create = schema.to_create_schema()
        assert create.file_url == "data:image/png;base64,dGVzdA=="


@pytest.mark.unit
class TestOcrTaskSchema:
    """Tests for OcrTaskSchema."""

    def test_webhook_exclude_fields(self) -> None:
        """webhook_exclude_fields should exclude result."""
        from apps.ocr.schemas import OcrTaskSchema

        schema = OcrTaskSchema(
            uid="task_123",
            user_id="user_123",
            file_url="https://example.com/file.pdf",
        )

        assert schema.webhook_exclude_fields == {"result"}
