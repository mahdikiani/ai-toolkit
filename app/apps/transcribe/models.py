"""Transcribe task model definition."""

from typing import Self

from fastapi_mongo_base.models import UserOwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import TranscribeTaskSchema


class TranscribeTask(UserOwnedEntity, TranscribeTaskSchema):  # type: ignore[misc]
    """Transcription task entity for converting audio to text."""

    async def start_processing(
        self, *, force_restart: bool = False, sync: bool = False, **kwargs: object
    ) -> Self:
        """Start processing the transcription task asynchronously."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_transcribe(  # type: ignore[return-value]
            self, force_restart=force_restart, sync=sync, **kwargs
        )
