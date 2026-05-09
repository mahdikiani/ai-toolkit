"""OCR task processing services."""

import logging

from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils import finance, mime, texttools

from .archive_services import process_compressed_archive
from .file_processors import (
    is_compressed_file,
    is_ocr_required,
)
from .models import OcrTask
from .no_ocr_services import process_direct_file
from .ocr_services import prepare_pages, process_pages_batch
from .paddle_ocr_services import process_pages_with_paddle
from .schemas import OcrEngineType


def _resolve_ocr_engine(task: OcrTask) -> OcrEngineType:
    """Resolve the OCR engine type from task configuration."""
    engine = (task.ocr_engine or Settings.ocr_engine or "llm").lower().strip()
    aliases = {
        "paddle": "paddleocr_vl_1_5",
        "paddleocr": "paddleocr_vl_1_5",
        "paddleocr_v1.5": "paddleocr_vl_1_5",
        "paddleocr_vl_1_5": "paddleocr_vl_1_5",
        "paddleocr-vl-1.5": "paddleocr_vl_1_5",
    }
    return OcrEngineType(aliases.get(engine, "llm"))


async def process_ocr(task: OcrTask) -> OcrTask:
    """Process an OCR task and extract text from the uploaded file."""
    try:
        file_content = await task.file_content()
        file_type = mime.check_file_type(file_content)

        # Compressed archive processing
        if is_compressed_file(file_type):
            return await process_compressed_archive(task, file_content, file_type)

        # Direct file processing (DOCX, PPTX)
        if not is_ocr_required(file_type):
            result = process_direct_file(file_content, file_type)
            return await save_result(task, result)

        # OCR processing (PDF, images)
        pages = prepare_pages(file_content, file_type)
        if not pages:
            return await save_error(
                task, f"Failed to prepare pages for file type: {file_type}"
            )

        # Check quota
        quota = await finance.check_quota(
            task.user_id, len(pages), raise_exception=False
        )
        if quota < len(pages):
            logging.error("Insufficient quota for task %s", task.uid)
            return await save_error(task, "insufficient_quota")

        # Process pages with OCR
        engine = _resolve_ocr_engine(task)
        if engine == "paddleocr_vl_1_5":
            text_pages = await process_pages_with_paddle(pages)
        else:
            text_pages = await process_pages_batch(pages, max_concurrent=10)

        # Meter usage
        usage = await finance.meter_cost(task.user_id, len(pages))

        # Save result
        result = "\n\n".join([t for t in text_pages if t])
        return await save_result(
            task,
            result,
            usage_amount=float(usage.amount) if usage else None,
            usage_id=usage.uid if usage else None,
        )

    except Exception:
        logging.exception("Error processing task %s", task.uid)
        return await save_error(task, "error")


async def save_error(task: OcrTask, message: str) -> OcrTask:
    """Save error result for a task."""
    task.task_status = TaskStatusEnum.error
    await task.save_report(message)
    return task


async def save_result(
    task: OcrTask,
    result: str,
    usage_amount: float | None = None,
    usage_id: str | None = None,
) -> OcrTask:
    """Save successful result for a task."""
    task.result = texttools.normalize_text(result)
    task.task_status = TaskStatusEnum.completed
    task.usage_amount = usage_amount
    task.usage_id = usage_id
    await task.save_report("Task processed successfully")
    return task
