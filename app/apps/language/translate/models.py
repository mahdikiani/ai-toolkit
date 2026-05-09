"""Translation task model definition."""

from typing import Self

from fastapi_mongo_base.models import UserOwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import TranslateSchema


class TranslateTask(UserOwnedEntity, TranslateSchema):  # type: ignore[misc]
    """Translation task entity for converting text between languages."""

    async def start_processing(self) -> Self:
        """Start processing the translation task asynchronously."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_translate(self)  # type: ignore[return-value]
