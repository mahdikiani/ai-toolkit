import logging

from fastapi_mongo_base.tasks import TaskStatusEnum

from utils import finance, mime, texttools

from .archive_services import process_compressed_archive
from .file_processors import (
    is_compressed_file,
    is_ocr_required,
)
from .models import OcrTask
from .no_ocr_services import process_direct_file
from .ocr_services import prepare_pages, process_pages_batch


async def process_ocr(task: OcrTask) -> OcrTask:
    """Main OCR processing function - simplified version."""
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
            return await save_error(task, "insufficient_quota")

        # Process pages with OCR
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
