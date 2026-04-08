import base64
import binascii
import json
from io import BytesIO

import httpx
from fastapi import Form, HTTPException
from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel, Field


class OcrTaskSchemaCreate(BaseModel):
    file_url: str = Field(
        min_length=1,
        description="The URL of the file or base64 encoded data to be OCRed",
    )
    user_id: str | None = Field(
        None, description="The ID of the user who is requesting the OCR (only admin)"
    )
    webhook_url: str | None = Field(
        None, description="The URL to send the OCR result to"
    )
    webhook_custom_headers: dict | None = Field(
        None, description="Custom headers to send with the OCR result"
    )
    meta_data: dict | None = Field(
        None, description="Additional metadata to be included in the OCR result"
    )

    @property
    def is_pdf(self) -> bool:
        return self.file_url.endswith(".pdf")

    async def file_content(self) -> BytesIO:
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

        async with httpx.AsyncClient() as client:
            response = await client.get(self.file_url)
            self._file_content.write(response.content)
            self._file_content.seek(0)
            return self._file_content

    async def file_content_base64(self) -> str:
        content = await self.file_content()
        return base64.b64encode(content.getvalue()).decode("utf-8")


class OcrTaskUploadFormSchema(BaseModel):
    user_id: str | None = None
    webhook_url: str | None = None
    webhook_custom_headers: dict | None = None
    meta_data: dict | None = None

    @classmethod
    def as_form(
        cls,
        user_id: str | None = Form(None),
        webhook_url: str | None = Form(None),
        webhook_custom_headers: str | None = Form(None),
        meta_data: str | None = Form(None),
    ) -> "OcrTaskUploadFormSchema":
        try:
            parsed_webhook_headers = (
                json.loads(webhook_custom_headers) if webhook_custom_headers else None
            )
            parsed_meta_data = json.loads(meta_data) if meta_data else None
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail="webhook_custom_headers and meta_data must be valid JSON.",
            ) from exc

        return cls(
            user_id=user_id,
            webhook_url=webhook_url,
            webhook_custom_headers=parsed_webhook_headers,
            meta_data=parsed_meta_data,
        )


class OcrTaskSchema(UserOwnedEntitySchema, TaskMixin, OcrTaskSchemaCreate):  # type: ignore[misc]
    result: str | None = None
    usage_amount: float | None = None
    usage_id: str | None = None

    @property
    def webhook_exclude_fields(self) -> set[str]:
        return {}
        return {"result"}
