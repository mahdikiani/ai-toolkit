"""YouTube transcription API routes for transcript task management."""

import json

from fastapi import BackgroundTasks, Query, Request
from fastapi.responses import PlainTextResponse, Response
from fastapi_mongo_base.routes import AbstractTaskRouter, PaginatedResponse
from fastapi_mongo_base.utils import usso_routes
from usso.integrations.fastapi import USSOAuthentication

from server.config import Settings

from .models import YoutubeTask
from .schemas import YoutubeTaskSchema, YoutubeTaskSchemaCreate


class YoutubeRouter(AbstractTaskRouter, usso_routes.AbstractTenantUSSORouter):
    """Router for YouTube transcription task management endpoints."""

    model = YoutubeTask
    schema = YoutubeTaskSchema

    def __init__(self) -> None:
        """Initialize the YouTube router with authentication and configuration."""
        super().__init__(
            user_dependency=USSOAuthentication(),
            draftable=False,
            prefix="/youtube",
            tags=["YouTube"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Configure YouTube-specific API routes."""
        super().config_routes(update_route=False, **kwargs)
        self.router.add_api_route(
            "/{uid}/result",
            self.get_result,
            methods=["GET"],
        )

    async def list_items(
        self,
        request: Request,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=Settings.page_max_limit),
        user_id: str | None = None,
    ) -> PaginatedResponse[YoutubeTaskSchema]:
        """List YouTube transcription tasks with pagination."""
        return await self._list_items(request, offset, limit, user_id=user_id)

    async def create_item(
        self,
        request: Request,
        data: YoutubeTaskSchemaCreate,
        background_tasks: BackgroundTasks,
        blocking: bool = False,
    ) -> YoutubeTask:
        """Create a new YouTube transcription task."""
        user = await self.get_user(request)
        data.user_id = data.user_id or user.user_id
        if data.user_id != user.user_id:
            await self.authorize(
                action="create", user=user, filter_data=data.model_dump()
            )

        item = await self.model.create_item({
            **data.model_dump(exclude_none=True),
            "tenant_id": user.tenant_id,
        })
        if blocking:
            await item.start_processing()
        else:
            background_tasks.add_task(item.start_processing)
        return item

    async def get_result(self, request: Request, uid: str):  # noqa: ANN201
        """Retrieve the result of a completed YouTube transcription task."""
        task: YoutubeTask = await self.retrieve_item(request, uid)

        if task.task_status != "completed":
            return PlainTextResponse(
                "No result available, please wait for the task to complete.",
            )

        return Response(
            content=json.dumps({"video_id": task.video_id, "transcript": task.result}),
            media_type="application/json",
        )


router = YoutubeRouter().router
