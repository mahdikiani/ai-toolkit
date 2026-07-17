"""YouTube transcription task model."""

from fastapi_mongo_base.models import UserOwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import YoutubeTranscriptTaskSchema


class YoutubeTranscriptTask(UserOwnedEntity, YoutubeTranscriptTaskSchema):
    """YouTube transcription task entity."""

    async def start_processing(self) -> "YoutubeTranscriptTask":
        """Start processing the YouTube transcription task."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_youtube(self)
