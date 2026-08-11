"""Conversion task Beanie model."""

from fastapi_mongo_base.models import TenantUserEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .task_schemas import ConversionTaskSchema


class ConversionTask(TenantUserEntity, ConversionTaskSchema):
    """Background conversion: Media URI → Artifacts → webhook."""

    async def start_processing(self) -> "ConversionTask":
        """Run from-media conversion pipeline."""
        from . import task_services

        self.task_status = TaskStatusEnum.processing
        await self.save()
        return await task_services.process_conversion_from_media(self)
