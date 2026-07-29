"""Web search task model."""

from fastapi_mongo_base.models import TenantUserEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import WebSearchTaskSchema


class WebSearchTask(TenantUserEntity, WebSearchTaskSchema):
    """Beanie document for a web search task."""

    async def start_processing(self) -> "WebSearchTask":
        """Run the web search for this task."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_search(self)
