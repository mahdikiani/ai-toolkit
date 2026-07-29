"""Imagination task model."""

from fastapi_mongo_base.models import TenantUserEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import ImaginationTaskSchema


class ImaginationTask(TenantUserEntity, ImaginationTaskSchema):
    """Beanie document for an image-generation task."""

    async def start_processing(self) -> "ImaginationTask":
        """Run image generation for this task."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_imagination(self)
