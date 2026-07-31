"""Webpage extraction API routes."""

from io import BytesIO

from fastapi import BackgroundTasks, Query, Request
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi_mongo_base.routes import PaginatedResponse
from pydantic import BaseModel

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from .models import WebpageTask
from .schemas import WebpageTaskSchema, WebpageTaskSchemaCreate


class WebpageRouter(AbstractTaskUSSORouter):
    """Router for webpage extraction task management endpoints."""

    model = WebpageTask
    schema = WebpageTaskSchema

    def __init__(self) -> None:
        """Initialize the webpage task router."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/webpages",
            tags=["Webpage"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Configure task routes and the result download endpoint."""
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
    ) -> PaginatedResponse[BaseModel]:
        """List webpage extraction tasks."""
        return await self._list_items(request, offset, limit, user_id=user_id)

    async def create_item(
        self,
        request: Request,
        data: WebpageTaskSchemaCreate,
        background_tasks: BackgroundTasks,
    ) -> WebpageTask:
        """Create and begin processing a webpage extraction task."""
        user = await self.get_user(request)
        await authorize_create_on_behalf(self, request, user, data)

        item: WebpageTask = await self.model.create_item({
            **data.model_dump(exclude_none=True),
            "tenant_id": user.tenant_id,
            "user_id": data.user_id or user.uid,
            "workspace_id": data.workspace_id,
        })
        background_tasks.add_task(item.start_processing)
        return item

    async def get_result(self, request: Request, uid: str) -> Response:
        """Return the completed extraction result as a text download."""
        task: WebpageTask = await self.retrieve_item(request, uid)

        if task.task_status != "completed":
            return PlainTextResponse(
                "No result available, please wait for the task to complete.",
            )

        return StreamingResponse(
            BytesIO((task.result or "").encode("utf-8")),
            media_type="text/plain",
            headers={"Content-Disposition": 'attachment; filename="result.txt"'},
        )


router = WebpageRouter().router
