"""YouTube transcription task model."""

from typing import Self

from fastapi_mongo_base.models import UserOwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import YoutubeTaskSchema


class YoutubeTask(UserOwnedEntity, YoutubeTaskSchema):  # type: ignore[misc]
    """YouTube transcription task entity."""

    async def start_processing(self) -> Self:
        """Start processing the YouTube transcription task."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_youtube(self)  # type: ignore[return-value]
