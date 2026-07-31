"""Video generation task model."""

from fastapi_mongo_base.models import TenantUserEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import VideoGenTaskSchema


class VideoGenTask(TenantUserEntity, VideoGenTaskSchema):
    """Video generation task entity."""

    async def start_processing(self) -> "VideoGenTask":
        """Start processing the video generation task."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_video(self)
