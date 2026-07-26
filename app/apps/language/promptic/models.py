"""Models for promptic task management."""

from typing import ClassVar

from fastapi_mongo_base.models import TenantUserEntity
from fastapi_mongo_base.tasks import TaskStatusEnum
from pymongo import ASCENDING, IndexModel

from .schemas import PrompticSchema


class PrompticTask(TenantUserEntity, PrompticSchema):
    """Promptic task for prompt template invocations."""

    class Settings(TenantUserEntity.Settings):
        """Beanie document settings with promptic indexes."""

        name = "promptic"
        indexes: ClassVar[list[IndexModel]] = [
            *TenantUserEntity.Settings.indexes,
            IndexModel([("idempotency_key", ASCENDING)]),
            IndexModel([("task_status", ASCENDING)]),
        ]

    @property
    def webhook_exclude_fields(self) -> set[str]:
        """Exclude large fields from webhook payload."""
        return {"input_variables", "task_logs", "webhook_custom_headers"}

    async def start_processing(
        self, *, force_restart: bool = False, sync: bool = False, **kwargs: object
    ) -> "PrompticTask":
        """Start background processing of the prompt template."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_promptic(
            self, force_restart=force_restart, sync=sync, **kwargs
        )
