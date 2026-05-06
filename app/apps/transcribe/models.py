"""Provide module functionality."""
from typing import Self

from fastapi_mongo_base.models import UserOwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import TranscribeTaskSchema


class TranscribeTask(UserOwnedEntity, TranscribeTaskSchema):  # type: ignore[misc]
    """Represent TranscribeTask."""

    async def start_processing(
        self, *, force_restart: bool = False, sync: bool = False, **kwargs: object
    ) -> Self:
        """Run start processing."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_transcribe(  # type: ignore[return-value]
            self, force_restart=force_restart, sync=sync, **kwargs
        )
