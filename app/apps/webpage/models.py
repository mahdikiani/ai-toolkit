"""Webpage extraction task model."""

from fastapi_mongo_base.models import UserOwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import WebpageTaskSchema


class WebpageTask(UserOwnedEntity, WebpageTaskSchema):
    """Webpage extraction task entity."""

    async def start_processing(self) -> "WebpageTask":
        """Start processing the webpage extraction task."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_webpage(self)
