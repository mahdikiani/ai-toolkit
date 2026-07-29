"""OCR task processing services."""

import asyncio
import logging
from io import BytesIO
from pathlib import Path

from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils import texttools
from utils.billing import finance
from utils.billing.saas import UsageSchema
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

logger = logging.getLogger(__name__)


async def _meter_usage(
    task: OcrTask, amount: float, meta_data: dict
) -> UsageSchema | None:
    """
    Record usage cost for a task; never let a billing failure erase a result.

    The OCR work (potentially many pages/minutes of VLM calls) has
    already happened by the time this runs -- a transient billing
    service outage must be logged, not allowed to propagate and mark an
    otherwise-successful task as failed.
    """
    try:
        return await finance.meter_cost(task.user_id, amount, meta_data=meta_data)
    except Exception:
        logger.exception("Failed to meter OCR usage for task %s", task.uid)
        return None


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
        "di": "document_intelligence",
        "document-intelligence": "document_intelligence",
        "ast": "document_intelligence",
    }
    return OcrEngineType(aliases.get(engine, engine))


_EXTENSION_BY_MIME = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


async def process_ocr(task: OcrTask) -> OcrTask:
    """
    Process an OCR task and extract text from the uploaded file.

    Engine selection (request field wins, else ``Settings.ocr_engine``
    / ``OCR_ENGINE``):

    - ``pipeline`` — layout detection + VLM element extraction
    - ``document_intelligence`` — AST pipeline with DOCX/OMML output
    - legacy: ``llm``, ``paddle`` / ``paddleocr_vl_1_5``
    """
    try:
        file_content = await task.file_content()
        file_type = mime.check_file_type(file_content)

        if is_compressed_file(file_type):
            return await process_compressed_archive(task, file_content, file_type)

        if not is_ocr_required(file_type):
            result = process_direct_file(file_content, file_type)
            return await save_result(task, result)

        engine = _resolve_ocr_engine(task)
        dispatch = {
            OcrEngineType.pipeline: _process_with_pipeline,
            OcrEngineType.document_intelligence: _process_with_document_intelligence,
        }
        handler = dispatch.get(engine)
        if handler is not None:
            return await handler(task, file_content, file_type, engine)

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
        usage = await _meter_usage(
            task,
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


async def _upload_pipeline_assets_and_docx(
    pipeline: object,
    result: str,
    file_content: BytesIO,
    file_type: str,
) -> tuple[str, str | None]:
    """Upload visual assets, rewrite placeholders, and build a DOCX upload URL."""
    from PIL import Image

    from utils.integrations.media import upload_file

    from .pipeline.docx_renderer import build_docx
    from .pipeline.renderer import render_pdf_bytes

    docx_url: str | None = None
    uploaded_assets: dict[str, str] = {}
    try:
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
        docx_buf = build_docx(
            result,
            page_images=page_images,
            elements=pipeline.get_elements(),
            page_headers=pipeline.get_headers(),
            page_footers=pipeline.get_footers(),
            crops=pipeline.get_all_crops(),
            pdf_data=pdf_data,
        )
        docx_buf.seek(0)
        docx_url = await upload_file(docx_buf)
    except Exception:
        logger.exception("DOCX generation / asset upload failed")
    return result, docx_url


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

    result, docx_url = await _upload_pipeline_assets_and_docx(
        pipeline, result, file_content, file_type
    )

    amount = finance.estimate_ocr_cost(pages=page_count, engine=engine.value)
    usage = await _meter_usage(
        task,
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


async def _process_with_document_intelligence(
    task: OcrTask,
    file_content: BytesIO,
    file_type: str,
    engine: OcrEngineType,
) -> OcrTask:
    """
    Process OCR using the AST-based Document Intelligence pipeline.

    Real per-element extraction (layout -> VLM -> reading order -> AST),
    rendered into clean Markdown plus a DOCX with real tables, OMML
    equations, and Word header/footer sections.
    """
    import shutil

    from .document_intelligence import DocumentIntelligencePipeline, summarize_stats
    from .document_intelligence.renderers.markdown import rewrite_asset_links
    from .pipeline.renderer import count_pdf_bytes

    page_count = count_pdf_bytes(file_content) if file_type == "application/pdf" else 1
    if page_count < 1:
        return await save_error(task, "Document has no pages")
    quota = await finance.check_quota(task.user_id, page_count, raise_exception=False)
    if quota < page_count:
        return await save_error(task, "insufficient_quota")

    filename = f"document{_EXTENSION_BY_MIME.get(file_type, '')}"
    di_pipeline = DocumentIntelligencePipeline()
    try:
        result = await di_pipeline.process(file_content, filename)
    except Exception as exc:
        di_pipeline.cleanup()
        shutil.rmtree(di_pipeline.output_dir, ignore_errors=True)
        logger.exception("Document Intelligence pipeline failed for task %s", task.uid)
        return await save_error(task, f"Document Intelligence pipeline failed: {exc}")
    di_pipeline.cleanup()

    markdown = result.markdown
    docx_url: str | None = None
    try:
        from utils.integrations.media import upload_file

        url_map: dict[str, str] = {}
        for asset in result.assets:
            try:
                content = await asyncio.to_thread(Path(asset.path).read_bytes)
                url = await upload_file(BytesIO(content))
                if url:
                    url_map[asset.rel_path] = url
            except Exception:
                logger.exception("Failed to upload asset %s", asset.rel_path)
        if url_map:
            markdown = rewrite_asset_links(markdown, url_map)

        docx_url = await upload_file(BytesIO(result.docx_bytes))
    except Exception:
        logger.exception("Asset/DOCX upload failed for task %s", task.uid)
    finally:
        shutil.rmtree(result.output_dir, ignore_errors=True)

    amount = finance.estimate_ocr_cost(pages=page_count, engine=engine.value)
    usage = await _meter_usage(
        task,
        amount,
        meta_data={"service": "ocr", "engine": engine.value, "pages": page_count},
    )

    provider_meta = {
        "provider": "ocr",
        "engine": engine.value,
        "pipeline": "document_intelligence_v1",
        "model": Settings.ocr_vlm_model,
        "usage": {"pages": page_count},
        "stats": summarize_stats(
            result.stats,
            include_elements=Settings.ocr_di_output_debug,
        ),
    }
    if docx_url:
        provider_meta["docx_url"] = docx_url

    return await save_result(
        task,
        markdown,
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
