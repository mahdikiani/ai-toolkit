"""Imagination API routes for image generation task management."""

from typing import cast

from fastapi import BackgroundTasks, Query, Request
from fastapi_mongo_base.routes import PaginatedResponse

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from .models import ImaginationTask
from .schemas import ImaginationTaskSchema, ImaginationTaskSchemaCreate


class ImaginationRouter(AbstractTaskUSSORouter):
    """Router for image-generation task endpoints."""

    model = ImaginationTask
    schema = ImaginationTaskSchema

    def __init__(self) -> None:
        """Configure the /imagination router prefix and tags."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/imagination",
            tags=["Imagination"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Register routes, excluding update/webhook (not applicable here)."""
        super().config_routes(update_route=False, webhook_route=False, **kwargs)

    async def list_items(
        self,
        request: Request,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=Settings.page_max_limit),
        user_id: str | None = None,
    ) -> PaginatedResponse[ImaginationTaskSchema]:
        """List imagination tasks, paginated."""
        return cast(
            PaginatedResponse[ImaginationTaskSchema],
            await self._list_items(request, offset, limit, user_id=user_id),
        )

    async def create_item(
        self,
        request: Request,
        data: ImaginationTaskSchemaCreate,
        background_tasks: BackgroundTasks,
    ) -> ImaginationTask:
        """Create an imagination task and start processing in the background."""
        user = await self.get_user(request)
        await authorize_create_on_behalf(self, request, user, data)

        item = await self.model.create_item({
            **data.model_dump(exclude_none=True),
            "tenant_id": user.tenant_id,
            "user_id": data.user_id or user.uid,
            "workspace_id": user.workspace_id,
        })
        background_tasks.add_task(item.start_processing)
        return item

    async def retrieve_item(self, request: Request, uid: str) -> ImaginationTask:
        """Retrieve a single imagination task by uid."""
        return await super().retrieve_item(request, uid)


router = ImaginationRouter().router
