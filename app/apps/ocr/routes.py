from io import BytesIO

from fastapi import BackgroundTasks, Query, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi_mongo_base.routes import AbstractTaskRouter, PaginatedResponse
from fastapi_mongo_base.utils import usso_routes
from usso.integrations.fastapi import USSOAuthentication

from server.config import Settings

from .models import OcrTask
from .schemas import OcrTaskSchema, OcrTaskSchemaCreate


class OCRRouter(AbstractTaskRouter, usso_routes.AbstractTenantUSSORouter):
    model = OcrTask
    schema = OcrTaskSchema

    def __init__(self) -> None:
        super().__init__(
            user_dependency=USSOAuthentication(),
            draftable=False,
            prefix="/ocrs",
            tags=["OCR"],
        )

    def config_routes(self, **kwargs: object) -> None:
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
    ) -> PaginatedResponse[OcrTaskSchema]:
        return await self._list_items(request, offset, limit, user_id=user_id)

    async def create_item(
        self,
        request: Request,
        data: OcrTaskSchemaCreate,
        background_tasks: BackgroundTasks,
        blocking: bool = False,
    ) -> OcrTask:
        user = await self.get_user(request)
        data.user_id = data.user_id or user.user_id
        if data.user_id != user.user_id:
            await self.authorize(
                action="create", user=user, filter_data=data.model_dump()
            )

        item = await self.model.create_item({
            **data.model_dump(exclude_none=True),
            "tenant_id": user.tenant_id,
            "task_status": "init",
        })
        if blocking:
            await item.start_processing()
        else:
            background_tasks.add_task(item.start_processing)
        return item

    async def get_result(self, request: Request, uid: str):  # noqa: ANN201
        task: OcrTask = await self.retrieve_item(request, uid)

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


router = OCRRouter().router
