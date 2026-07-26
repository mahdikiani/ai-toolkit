"""YouTube transcription task model."""

from fastapi_mongo_base.models import TenantUserEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import YoutubeTranscriptTaskSchema


class YoutubeTranscriptTask(TenantUserEntity, YoutubeTranscriptTaskSchema):
    """YouTube transcription task entity."""

    async def start_processing(self) -> "YoutubeTranscriptTask":
        """Start processing the YouTube transcription task."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_youtube(self)
