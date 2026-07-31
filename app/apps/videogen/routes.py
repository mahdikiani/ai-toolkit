"""Video generation API routes for video task management."""

from typing import cast

from fastapi import BackgroundTasks, Query, Request
from fastapi_mongo_base.routes import PaginatedResponse

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from .models import VideoGenTask
from .schemas import VideoGenTaskSchema, VideoGenTaskSchemaCreate


class VideoGenRouter(AbstractTaskUSSORouter):
    """Router for video generation task management endpoints."""

    model = VideoGenTask
    schema = VideoGenTaskSchema

    def __init__(self) -> None:
        """Initialize the video generation router with auth and configuration."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/video",
            tags=["Video Generation"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Configure video generation-specific API routes."""
        super().config_routes(update_route=False, webhook_route=False, **kwargs)

    async def list_items(
        self,
        request: Request,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=Settings.page_max_limit),
        user_id: str | None = None,
    ) -> PaginatedResponse[VideoGenTaskSchema]:
        """List video generation tasks with pagination."""
        return cast(
            PaginatedResponse[VideoGenTaskSchema],
            await self._list_items(request, offset, limit, user_id=user_id),
        )

    async def create_item(
        self,
        request: Request,
        data: VideoGenTaskSchemaCreate,
        background_tasks: BackgroundTasks,
    ) -> VideoGenTask:
        """Create a new video generation task."""
        user = await self.get_user(request)
        await authorize_create_on_behalf(self, request, user, data)
        item = await self.model.create_item({
            **data.model_dump(exclude_none=True),
            "tenant_id": user.tenant_id,
            "user_id": data.user_id or user.uid,
            "workspace_id": data.workspace_id,
        })
        background_tasks.add_task(item.start_processing)
        return item


router = VideoGenRouter().router
