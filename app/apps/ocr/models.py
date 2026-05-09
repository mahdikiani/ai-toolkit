"""OCR task model definition."""

from typing import Self

from fastapi_mongo_base.models import UserOwnedEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import OcrTaskSchema


class OcrTask(UserOwnedEntity, OcrTaskSchema):  # type: ignore[misc]
    """OCR task entity for processing document text extraction."""

    async def start_processing(self) -> Self:
        """Start processing the OCR task asynchronously."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_ocr(self)  # type: ignore[return-value]
