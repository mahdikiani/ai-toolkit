"""Provide module functionality."""
from typing import Self

from fastapi_mongo_base.models import UserOwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import TranslateSchema


class TranslateTask(UserOwnedEntity, TranslateSchema):  # type: ignore[misc]
    """Represent TranslateTask."""

    async def start_processing(self) -> Self:
        """Run start processing."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_translate(self)  # type: ignore[return-value]
