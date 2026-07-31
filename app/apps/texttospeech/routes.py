"""Text-to-speech API routes for TTS task management."""

import base64
from typing import cast

from fastapi import BackgroundTasks, Query, Request
from fastapi_mongo_base.routes import PaginatedResponse

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from .models import TextToSpeechTask
from .schemas import TextToSpeechTaskSchema, TextToSpeechTaskSchemaCreate


class TextToSpeechRouter(AbstractTaskUSSORouter):
    """Router for text-to-speech task management endpoints."""

    model = TextToSpeechTask
    schema = TextToSpeechTaskSchema

    def __init__(self) -> None:
        """Initialize the text-to-speech router with auth and configuration."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/text-to-speech",
            tags=["Text to Speech"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Configure text-to-speech-specific API routes."""
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
    ) -> PaginatedResponse[TextToSpeechTaskSchema]:
        """List text-to-speech tasks with pagination."""
        return cast(
            PaginatedResponse[TextToSpeechTaskSchema],
            await self._list_items(request, offset, limit, user_id=user_id),
        )

    async def create_item(
        self,
        request: Request,
        data: TextToSpeechTaskSchemaCreate,
        background_tasks: BackgroundTasks,
    ) -> TextToSpeechTask:
        """Create a new text-to-speech task."""
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
        """Retrieve the result of a completed text-to-speech task."""
        task: TextToSpeechTask = await self.retrieve_item(request, uid)
        if task.task_status != "completed":
            return {"status": task.task_status, "message": "Not yet complete"}
        audio_data = task.result_data
        b64 = base64.b64encode(audio_data).decode() if audio_data else None
        return {
            "status": "completed",
            "text": task.text,
            "audio_base64": b64,
            "format": task.response_format,
        }


router = TextToSpeechRouter().router
