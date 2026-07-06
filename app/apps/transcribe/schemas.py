"""Transcribe task schemas and data models."""

import base64
from io import BytesIO

import httpx
from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from fastapi_mongo_base.tasks import TaskMixin
from pydantic import BaseModel, Field


class TranscribeTaskSchemaCreate(BaseModel):
    """Schema for creating a new transcription task."""

    file_url: str
    audio_duration_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Known audio duration in seconds, if the client already has it.",
    )
    provider: str = "soniox"
    model: str | None = None
    user_id: str | None = None
    webhook_url: str | None = None

    async def file_content(self) -> BytesIO:
        """Fetch and return audio file content from URL."""
        if hasattr(self, "_file_content"):
            return getattr(self, "_file_content", BytesIO())

        self._file_content = BytesIO()
        async with httpx.AsyncClient() as client:
            response = await client.get(self.file_url)
            self._file_content.write(response.content)
            self._file_content.seek(0)
            return self._file_content

    async def file_content_base64(self) -> str:
        """Return audio file content encoded as base64 string."""
        content = await self.file_content()
        return base64.b64encode(content.getvalue()).decode("utf-8")


class ChunkMetadata(BaseModel):
    """Metadata for a transcribed audio chunk."""

    chunk_id: int
    start_ms: int
    end_ms: int
    file_path: str
    job_id: str | None = None
    text: str | None = None


class TranscribeTaskSchema(  # type: ignore[misc]
    UserOwnedEntitySchema, TaskMixin, TranscribeTaskSchemaCreate
):
    """Complete transcription task schema including result and chunk metadata."""

    result: str | None = None
    usage_amount: float | None = None
    usage_id: str | None = None
    provider_meta: dict | None = None
    transcription_job_id: str | None = None
    chunks: list[ChunkMetadata] | None = None

    @property
    def audio_duration(self) -> float:
        """Return known audio duration in seconds without network/file I/O."""
        if self.audio_duration_seconds is not None:
            return self.audio_duration_seconds

        if self.provider_meta:
            usage = self.provider_meta.get("usage")
            if isinstance(usage, dict):
                seconds = usage.get("audio_duration_seconds")
                if isinstance(seconds, int | float):
                    return float(seconds)

        if self.chunks:
            return max(chunk.end_ms for chunk in self.chunks) / 1000

        meta_data = self.meta_data or {}
        duration = meta_data.get("audio_duration_seconds")
        if isinstance(duration, int | float):
            return float(duration)

        duration_ms = meta_data.get("audio_duration_ms")
        if isinstance(duration_ms, int | float):
            return float(duration_ms) / 1000

        return 0.0
