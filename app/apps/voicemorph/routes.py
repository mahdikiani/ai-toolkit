"""Voice morphing API routes for voice morph task management."""

from typing import cast

from fastapi import BackgroundTasks, Query, Request
from fastapi_mongo_base.routes import PaginatedResponse

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from .models import VoiceMorphTask
from .schemas import VoiceMorphTaskSchema, VoiceMorphTaskSchemaCreate


class VoiceMorphRouter(AbstractTaskUSSORouter):
    """Router for voice morph task management endpoints."""

    model = VoiceMorphTask
    schema = VoiceMorphTaskSchema

    def __init__(self) -> None:
        """Initialize the voice morph router with auth and configuration."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/voice-morph",
            tags=["Voice Morph"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Configure voice morph-specific API routes."""
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
    ) -> PaginatedResponse[VoiceMorphTaskSchema]:
        """List voice morph tasks with pagination."""
        return cast(
            PaginatedResponse[VoiceMorphTaskSchema],
            await self._list_items(request, offset, limit, user_id=user_id),
        )

    async def create_item(
        self,
        request: Request,
        data: VoiceMorphTaskSchemaCreate,
        background_tasks: BackgroundTasks,
    ) -> VoiceMorphTask:
        """Create a new voice morph task."""
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

    async def get_result(self, request: Request, uid: str) -> dict:
        """Retrieve the result of a completed voice morph task."""
        task: VoiceMorphTask = await self.retrieve_item(request, uid)
        if task.task_status != "completed":
            return {"status": task.task_status, "message": "Not yet complete"}
        return {
            "status": "completed",
            "result_url": task.result_url,
        }


router = VoiceMorphRouter().router
