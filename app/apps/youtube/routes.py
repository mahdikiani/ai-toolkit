"""YouTube transcription API routes for transcript task management."""

import json
from typing import cast

from fastapi import BackgroundTasks, Query, Request
from fastapi.responses import PlainTextResponse, Response
from fastapi_mongo_base.routes import PaginatedResponse

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from .models import YoutubeTranscriptTask
from .schemas import YoutubeTranscriptTaskSchema, YoutubeTranscriptTaskSchemaCreate


class YoutubeRouter(AbstractTaskUSSORouter):
    """Router for YouTube transcription task management endpoints."""

    model = YoutubeTranscriptTask
    schema = YoutubeTranscriptTaskSchema

    def __init__(self) -> None:
        """Initialize the YouTube router with authentication and configuration."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/youtube",
            tags=["YouTube"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Configure YouTube-specific API routes."""
        super().config_routes(update_route=False, webhook_route=False, **kwargs)
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
    ) -> PaginatedResponse[YoutubeTranscriptTaskSchema]:
        """List YouTube transcription tasks with pagination."""
        return cast(
            PaginatedResponse[YoutubeTranscriptTaskSchema],
            await self._list_items(request, offset, limit, user_id=user_id),
        )

    async def create_item(
        self,
        request: Request,
        data: YoutubeTranscriptTaskSchemaCreate,
        background_tasks: BackgroundTasks,
    ) -> YoutubeTranscriptTask:
        """Create a new YouTube transcription task."""
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

    async def get_result(self, request: Request, uid: str) -> Response:
        """Retrieve the result of a completed YouTube transcription task."""
        task: YoutubeTranscriptTask = await self.retrieve_item(request, uid)

        if task.task_status != "completed":
            return PlainTextResponse(
                "No result available, please wait for the task to complete.",
            )

        return Response(
            content=json.dumps({"video_id": task.video_id, "transcript": task.result}),
            media_type="application/json",
        )


router = YoutubeRouter().router
