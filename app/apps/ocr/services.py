"""OCR task processing services."""

import logging
from io import BytesIO

logger = logging.getLogger(__name__)

from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils import texttools
from utils.billing import finance
from utils.files import mime

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
    engine = (task.ocr_engine or Settings.ocr_engine or "pipeline").lower().strip()
    aliases = {
        "paddle": "paddleocr_vl_1_5",
        "paddleocr": "paddleocr_vl_1_5",
        "paddleocr_v1.5": "paddleocr_vl_1_5",
        "paddleocr_vl_1_5": "paddleocr_vl_1_5",
        "paddleocr-vl-1.5": "paddleocr_vl_1_5",
        "modern": "pipeline",
        "layout": "pipeline",
    }
    return OcrEngineType(aliases.get(engine, engine))


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

        engine = _resolve_ocr_engine(task)

        # Modern pipeline engine: layout detection + structured extraction
        if engine == OcrEngineType.pipeline:
            return await _process_with_pipeline(task, file_content, file_type, engine)

        # Legacy engines (LLM, Paddle)
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

        if engine in (OcrEngineType.paddle, OcrEngineType.paddleocr_vl_1_5):
            text_pages = await process_pages_with_paddle(pages)
        else:
            text_pages = await process_pages_batch(pages, max_concurrent=10)

        amount = finance.estimate_ocr_cost(pages=len(pages), engine=engine.value)
        usage = await finance.meter_cost(
            task.user_id,
            amount,
            meta_data={
                "service": "ocr",
                "engine": engine.value,
                "pages": len(pages),
            },
        )

        result = "\n\n".join([t for t in text_pages if t])
        return await save_result(
            task,
            result,
            usage_amount=float(usage.amount) if usage else None,
            usage_id=usage.uid if usage else None,
            provider_meta={
                "provider": "ocr",
                "engine": engine.value,
                "usage": {"pages": len(pages)},
            },
        )

    except Exception as exc:
        logging.exception("Error processing task %s", task.uid)
        return await save_error(task, f"OCR processing failed: {exc}")


async def _process_with_pipeline(
    task: OcrTask,
    file_content: BytesIO,
    file_type: str,
    engine: OcrEngineType,
) -> OcrTask:
    """Process OCR using the modern document pipeline."""
    from .pipeline.engine import DocumentPipeline
    from .pipeline.layout_detector import LayoutBox
    from .pipeline.renderer import count_pdf_bytes

    async def ocr_fn(crop_bytes: BytesIO, element: LayoutBox) -> str:
        from .ocr_services import ocr_to_text

        return await ocr_to_text(crop_bytes, block_type=element.type.value)

    page_count = count_pdf_bytes(file_content) if file_type == "application/pdf" else 1
    if page_count < 1:
        return await save_error(task, "Document has no pages")
    quota = await finance.check_quota(task.user_id, page_count, raise_exception=False)
    if quota < page_count:
        return await save_error(task, "insufficient_quota")

    pipeline = DocumentPipeline(
        dpi=getattr(Settings, "ocr_pipeline_dpi", 300),
        enable_preprocessing=Settings.ocr_pipeline_enable_preprocessing,
        enable_layout=Settings.ocr_pipeline_enable_layout,
        enable_normalization=True,
        pipeline_ocr_fn=ocr_fn,
    )

    if file_type == "application/pdf":
        result = await pipeline.process_pdf(file_content)
    else:
        result = await pipeline.process_image_bytes(file_content)

    # Upload accumulated visual assets and replace asset:ID placeholders
    docx_url: str | None = None
    uploaded_assets: dict[str, str] = {}
    try:
        from .pipeline.docx_renderer import build_docx
        from .pipeline.renderer import render_pdf_bytes
        from PIL import Image
        from utils.integrations.media import upload_file

        assets = pipeline.get_assets()
        for asset in assets:
            try:
                buf = BytesIO(asset["image_bytes"])
                buf.seek(0)
                url = await upload_file(buf)
                if url:
                    uploaded_assets[asset["id"]] = url
            except Exception:
                logger.exception("Failed to upload asset %s", asset["id"])

        if uploaded_assets:
            for asset_id, url in uploaded_assets.items():
                result = result.replace(f"({asset_id})", f"({url})")

        page_images: list[Image.Image] = []
        if file_type == "application/pdf":
            file_content.seek(0)
            page_images = render_pdf_bytes(file_content, dpi=150)
        file_content.seek(0)
        pdf_data = file_content.read() if file_type == "application/pdf" else None
        docx_buf = build_docx(result, page_images, pdf_data=pdf_data)
        docx_buf.seek(0)
        docx_url = await upload_file(docx_buf)
    except Exception:
        logger.exception("DOCX generation / asset upload failed")

    amount = finance.estimate_ocr_cost(pages=page_count, engine=engine.value)
    usage = await finance.meter_cost(
        task.user_id,
        amount,
        meta_data={
            "service": "ocr",
            "engine": engine.value,
            "pages": page_count,
        },
    )

    provider_meta = {
        "provider": "ocr",
        "engine": engine.value,
        "pipeline": "document_pipeline_v1",
        "model": Settings.ocr_vlm_model,
        "usage": {"pages": page_count},
    }
    if docx_url:
        provider_meta["docx_url"] = docx_url

    return await save_result(
        task,
        result,
        usage_amount=float(usage.amount) if usage else None,
        usage_id=usage.uid if usage else None,
        provider_meta=provider_meta,
    )


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
    provider_meta: dict | None = None,
) -> OcrTask:
    """Save successful result for a task."""
    task.result = texttools.normalize_text(result)
    task.task_status = TaskStatusEnum.completed
    task.usage_amount = usage_amount
    task.usage_id = usage_id
    task.provider_meta = provider_meta
    await task.save_report("Task processed successfully")
    return task
