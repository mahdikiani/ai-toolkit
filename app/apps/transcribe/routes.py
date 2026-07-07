"""Transcribe API routes for audio transcription task management."""

import base64
from io import BytesIO

from fastapi import BackgroundTasks, Depends, File, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi_mongo_base.routes import AbstractTaskRouter, PaginatedResponse
from fastapi_mongo_base.utils import usso_routes
from soniox.types import TranscriptionWebhook
from usso.integrations.fastapi import USSOAuthentication

from server.config import Settings
from utils import speechmatics

from . import services
from .models import TranscribeTask
from .schemas import (
    TranscribeTaskBase64Schema,
    TranscribeTaskSchema,
    TranscribeTaskSchemaCreate,
    TranscribeTaskUploadFormSchema,
)


class TranscribeRouter(AbstractTaskRouter, usso_routes.AbstractTenantUSSORouter):
    """Router for transcription task management endpoints."""

    model = TranscribeTask
    schema = TranscribeTaskSchema

    def __init__(self) -> None:
        """Initialize the transcribe router with authentication and configuration."""
        super().__init__(
            user_dependency=USSOAuthentication(),
            draftable=False,
            prefix="/transcribes",
            tags=["Transcribe"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Configure transcription-specific API routes."""
        super().config_routes(update_route=False, **kwargs)
        self.router.add_api_route(
            "/{uid}/webhook",
            self.webhook,
            methods=["POST"],
            status_code=200,
        )
        self.router.add_api_route(
            "/{uid}/webhook/{chunk_id}",
            self.webhook_chunk,
            methods=["POST"],
            status_code=200,
        )
        self.router.add_api_route(
            "/upload/file",
            self.create_item_with_upload,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/upload/base64",
            self.create_item_with_base64,
            methods=["POST"],
        )
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
    ) -> PaginatedResponse[TranscribeTaskSchema]:
        """List transcription tasks with pagination."""
        return await self._list_items(request, offset, limit, user_id=user_id)

    async def create_item(
        self,
        request: Request,
        data: TranscribeTaskSchemaCreate,
        background_tasks: BackgroundTasks,
        blocking: bool = False,
    ) -> TranscribeTask:
        """Create a new transcription task from a file URL."""
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
        """Retrieve the result of a completed transcription task."""
        task: TranscribeTask = await self.retrieve_item(request, uid)

        # Assuming the OCR result is stored in task.result or similar
        # Adjust the attribute as per your OcrTask model
        if task.task_status != "completed":
            return PlainTextResponse(
                "No result available, please wait for the task to complete.",
            )

        return StreamingResponse(
            BytesIO((task.result or "").encode("utf-8")),
            media_type="text/plain",
            headers={"Content-Disposition": 'attachment; filename="result.txt"'},
        )

    async def create_item_with_upload(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        data_form: TranscribeTaskUploadFormSchema = Depends(
            TranscribeTaskUploadFormSchema.as_form
        ),
        blocking: bool = Query(False),
    ) -> TranscribeTask:
        """Create a transcription task from a direct multipart upload."""
        file_content = await file.read()
        encoded_file = base64.b64encode(file_content).decode("utf-8")
        mime_type = file.content_type or "application/octet-stream"
        data = TranscribeTaskSchemaCreate(
            file_url=f"data:{mime_type};base64,{encoded_file}",
            audio_duration_seconds=data_form.audio_duration_seconds,
            provider=data_form.provider,
            model=data_form.model,
            user_id=data_form.user_id,
            webhook_url=data_form.webhook_url,
        )
        return await self.create_item(request, data, background_tasks, blocking)

    async def create_item_with_base64(
        self,
        request: Request,
        data: TranscribeTaskBase64Schema,
        background_tasks: BackgroundTasks,
        blocking: bool = Query(False),
    ) -> TranscribeTask:
        """Create a transcription task from a base64 encoded payload."""
        return await self.create_item(
            request,
            data.to_create_schema(),
            background_tasks,
            blocking,
        )

    async def webhook(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
        uid: str,
        data: speechmatics.TranscribeWebhookSchema | TranscriptionWebhook | None = None,
        status: str | None = None,
    ) -> dict:
        """Handle transcription completion webhook."""
        item: TranscribeTask = await self.get_item(
            uid, user_id=None, ignore_user_id=True
        )
        if status == "error":
            background_tasks.add_task(services.process_error_webhook, item)
            return {"message": "Error"}

        if isinstance(data, TranscriptionWebhook):
            background_tasks.add_task(
                services.process_transcription_webhook, item, data
            )
        else:
            await services.save_error(item, "Invalid webhook data")
        return {}

    async def webhook_chunk(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
        uid: str,
        chunk_id: int,
        data: speechmatics.TranscribeWebhookSchema | TranscriptionWebhook | None = None,
        status: str | None = None,
    ) -> dict:
        """Handle chunk transcription webhook."""
        item: TranscribeTask = await self.get_item(
            uid, user_id=None, ignore_user_id=True
        )
        if status == "error":
            background_tasks.add_task(services.process_error_webhook, item)
            return {"message": "Error"}

        if isinstance(data, TranscriptionWebhook):
            background_tasks.add_task(
                services.process_transcription_webhook, item, data
            )
        else:
            await services.save_error(item, "Invalid webhook data")
        return {}


router = TranscribeRouter().router
