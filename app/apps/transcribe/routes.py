from io import BytesIO

from fastapi import BackgroundTasks, Query, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi_mongo_base.routes import AbstractTaskRouter, PaginatedResponse
from fastapi_mongo_base.utils import usso_routes
from soniox.types import TranscriptionWebhook
from usso.integrations.fastapi import USSOAuthentication

from server.config import Settings
from utils import speechmatics

from . import services
from .models import TranscribeTask
from .schemas import TranscribeTaskSchema, TranscribeTaskSchemaCreate


class TranscribeRouter(AbstractTaskRouter, usso_routes.AbstractTenantUSSORouter):
    model = TranscribeTask
    schema = TranscribeTaskSchema

    def __init__(self) -> None:
        super().__init__(
            user_dependency=USSOAuthentication(),
            draftable=False,
            prefix="/transcribes",
            tags=["Transcribe"],
        )

    def config_routes(self, **kwargs: object) -> None:
        super().config_routes(update_route=False, **kwargs)
        self.router.add_api_route(
            "/{uid}/webhook",
            self.webhook,
            methods=["POST"],
            status_code=200,
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
        return await self._list_items(request, offset, limit, user_id=user_id)

    async def create_item(
        self,
        request: Request,
        data: TranscribeTaskSchemaCreate,
        background_tasks: BackgroundTasks,
        blocking: bool = False,
    ) -> TranscribeTask:
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

    async def webhook(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
        uid: str,
        data: speechmatics.TranscribeWebhookSchema | TranscriptionWebhook | None = None,
        status: str | None = None,
    ) -> dict:
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
