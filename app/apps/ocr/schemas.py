"""OCR task schemas and data models."""

import base64
import binascii
import json
from enum import StrEnum
from io import BytesIO

from fastapi import Form
from fastapi_mongo_base.core.exceptions import BaseHTTPException
from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin, TaskMixin
from pydantic import Field

from utils.downloaders import download_bytes


class OcrEngineType(StrEnum):
    """Supported OCR engine types."""

    llm = "llm"
    paddle = "paddle"
    paddleocr_vl_1_5 = "paddleocr_vl_1_5"
    pipeline = "pipeline"


class OcrTaskSchemaCreate(TaskCreateFieldsMixin):
    """Schema for creating a new OCR task."""

    file_url: str = Field(
        min_length=1,
        description="The URL of the file or base64 encoded data to be OCRed",
    )
    webhook_custom_headers: dict | None = Field(
        None, description="Custom headers to send with the OCR result"
    )
    ocr_engine: OcrEngineType | None = Field(
        None,
        description=(
            "OCR engine: pipeline (default), llm, or paddleocr_vl_1_5. "
            "When omitted, server default is used."
        ),
    )

    @property
    def is_pdf(self) -> bool:
        """Check if the file URL points to a PDF file."""
        return self.file_url.endswith(".pdf")

    async def file_content(self) -> BytesIO:
        """Fetch and return file content from URL or data URL."""
        if hasattr(self, "_file_content"):
            return getattr(self, "_file_content", BytesIO())

        self._file_content = BytesIO()
        if self.file_url.startswith("data:"):
            _, _, encoded_payload = self.file_url.partition(",")
            try:
                self._file_content.write(base64.b64decode(encoded_payload))
            except binascii.Error:
                self._file_content.seek(0)
                return self._file_content
            self._file_content.seek(0)
            return self._file_content

        self._file_content = await download_bytes(self.file_url)
        return self._file_content

    async def file_content_base64(self) -> str:
        """Return file content encoded as base64 string."""
        content = await self.file_content()
        return base64.b64encode(content.getvalue()).decode("utf-8")


class OcrTaskUploadFormSchema(TaskCreateFieldsMixin):
    """Schema for multipart form upload of OCR tasks."""

    webhook_custom_headers: dict | None = None
    ocr_engine: OcrEngineType | None = None

    @classmethod
    def as_form(
        cls,
        user_id: str | None = Form(None),
        webhook_url: str | None = Form(None),
        webhook_custom_headers: str | None = Form(None),
        meta_data: str | None = Form(None),
        ocr_engine: str | None = Form(None),
    ) -> "OcrTaskUploadFormSchema":
        """Parse form fields into a validated schema instance."""
        try:
            parsed_webhook_headers = (
                json.loads(webhook_custom_headers) if webhook_custom_headers else None
            )
            parsed_meta_data = json.loads(meta_data) if meta_data else None
        except json.JSONDecodeError as exc:
            msg = "webhook_custom_headers and meta_data must be valid JSON."
            raise BaseHTTPException(
                status_code=422,
                error_code="invalid_json",
                detail=msg,
                message={"en": msg},
            ) from exc

        return cls(
            user_id=user_id,
            webhook_url=webhook_url,
            webhook_custom_headers=parsed_webhook_headers,
            meta_data=parsed_meta_data,
            ocr_engine=ocr_engine,
        )


class OcrTaskBase64Schema(TaskCreateFieldsMixin):
    """Base64 upload payload for OCR tasks."""

    content_base64: str = Field(..., min_length=1)
    mime_type: str = "application/octet-stream"
    webhook_custom_headers: dict | None = None
    ocr_engine: OcrEngineType | None = None

    def to_create_schema(self) -> OcrTaskSchemaCreate:
        """OCR create schema represented as a data URL."""
        encoded_payload = self.content_base64.strip()
        file_url = (
            encoded_payload
            if encoded_payload.startswith("data:")
            else f"data:{self.mime_type};base64,{encoded_payload}"
        )
        return OcrTaskSchemaCreate(
            file_url=file_url,
            user_id=self.user_id,
            webhook_url=self.webhook_url,
            webhook_custom_headers=self.webhook_custom_headers,
            meta_data=self.meta_data,
            ocr_engine=self.ocr_engine,
        )


class OcrTaskSchema(UserOwnedEntitySchema, TaskMixin, OcrTaskSchemaCreate):
    """Complete OCR task schema including result and usage fields."""

    result: str | None = None
    usage_amount: float | None = None
    usage_id: str | None = None
    provider_meta: dict | None = None

    @property
    def webhook_exclude_fields(self) -> set[str]:
        """Fields excluded from webhook payloads."""
        return {"result"}
