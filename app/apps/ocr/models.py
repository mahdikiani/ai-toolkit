"""OCR task model definition."""

from fastapi_mongo_base.models import TenantUserEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import OcrTaskSchema


class OcrTask(TenantUserEntity, OcrTaskSchema):
    """OCR task entity for processing document text extraction."""

    async def start_processing(self) -> "OcrTask":
        """Start processing the OCR task asynchronously."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_ocr(self)
