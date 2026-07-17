"""Translation task model definition."""

from fastapi_mongo_base.models import UserOwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import TranslateSchema


class TranslateTask(UserOwnedEntity, TranslateSchema):
    """Translation task entity for converting text between languages."""

    async def start_processing(self) -> "TranslateTask":
        """Start processing the translation task asynchronously."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_translate(self)
