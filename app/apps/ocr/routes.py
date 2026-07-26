"""OCR API routes for task management and file uploads."""

import base64
from io import BytesIO

from fastapi import (
    BackgroundTasks,
    Depends,
    File,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi_mongo_base.routes import PaginatedResponse
from pydantic import BaseModel

from server.config import Settings
from utils.auth import authorize_create_on_behalf
from utils.task_routes import AbstractTaskUSSORouter

from .models import OcrTask
from .schemas import (
    OcrTaskBase64Schema,
    OcrTaskSchema,
    OcrTaskSchemaCreate,
    OcrTaskUploadFormSchema,
)


class OCRRouter(AbstractTaskUSSORouter):
    """Router for OCR task management endpoints."""

    model = OcrTask
    schema = OcrTaskSchema

    def __init__(self) -> None:
        """Initialize the OCR router with authentication and configuration."""
        super().__init__(
            user_dependency=None,
            draftable=False,
            prefix="/ocrs",
            tags=["OCR"],
        )

    def config_routes(self, **kwargs: object) -> None:
        """Configure OCR-specific API routes."""
        super().config_routes(update_route=False, webhook_route=False, **kwargs)
        self.router.add_api_route(
            "/{uid:str}/start",
            self.start_item,
            methods=["POST"],
            response_model=self.retrieve_response_schema,
        )
        self.router.add_api_route(
            "/{uid}/result",
            self.get_result,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/upload",
            self.create_item_with_upload,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/upload/base64",
            self.create_item_with_base64,
            methods=["POST"],
        )

    async def list_items(
        self,
        request: Request,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=Settings.page_max_limit),
        user_id: str | None = None,
    ) -> PaginatedResponse[BaseModel]:
        """List OCR tasks with pagination."""
        return await self._list_items(request, offset, limit, user_id=user_id)

    async def create_item(
        self,
        request: Request,
        data: OcrTaskSchemaCreate,
        background_tasks: BackgroundTasks,
    ) -> OcrTask:
        """Create a new OCR task from a file URL."""
        user = await self.get_user(request)
        await authorize_create_on_behalf(self, request, user, data)

        item: OcrTask = await self.model.create_item({
            **data.model_dump(exclude_none=True),
            "tenant_id": user.tenant_id,
            "user_id": data.user_id or user.uid,
        })
        background_tasks.add_task(item.start_processing)
        return item

    async def get_result(self, request: Request, uid: str) -> Response:
        """Retrieve the result of a completed OCR task."""
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

    async def create_item_with_upload(
        self,
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        data_form: OcrTaskUploadFormSchema = Depends(OcrTaskUploadFormSchema.as_form),
    ) -> OcrTask:
        """Create an OCR task by directly uploading a file."""
        file_content = await file.read()
        encoded_file = base64.b64encode(file_content).decode("utf-8")
        mime_type = file.content_type or "application/octet-stream"
        file_url = f"data:{mime_type};base64,{encoded_file}"

        data = OcrTaskSchemaCreate(
            file_url=file_url,
            user_id=data_form.user_id,
            webhook_url=data_form.webhook_url,
            webhook_custom_headers=data_form.webhook_custom_headers,
            meta_data=data_form.meta_data,
            ocr_engine=data_form.ocr_engine,
        )
        return await self.create_item(
            request=request,
            data=data,
            background_tasks=background_tasks,
        )

    async def create_item_with_base64(
        self,
        request: Request,
        data: OcrTaskBase64Schema,
        background_tasks: BackgroundTasks,
    ) -> OcrTask:
        """Create an OCR task from a base64 encoded payload."""
        return await self.create_item(
            request=request,
            data=data.to_create_schema(),
            background_tasks=background_tasks,
        )


router = OCRRouter().router
