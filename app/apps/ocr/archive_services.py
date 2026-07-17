"""Services for processing compressed archives for OCR."""

import logging
import shutil
from io import BytesIO
from pathlib import Path

from anyio import Path as AsyncPath

from utils.billing import finance
from utils.files import archive_utils, mime, pdftools
from utils.integrations import media

from .file_processors import CONVERTING_IMAGE_EXTS, IMAGE_EXTS, is_ocr_required
from .models import OcrTask
from .no_ocr_services import process_direct_file
from .ocr_services import prepare_pages


def get_pages(file_path: Path) -> int:
    """Get number of pages in file."""
    with open(file_path, "rb") as file:
        file_content = BytesIO(file.read())
    file_type = mime.check_file_type(file_content)
    if file_type == "application/pdf":
        return pdftools.number_of_pages(file_path)
    elif file_type in CONVERTING_IMAGE_EXTS | IMAGE_EXTS:
        return 1
    else:
        return 0


async def process_file(file_path: Path) -> str | None:
    """Process file and return extracted text."""
    from .services import process_pages_batch

    # Direct file processing (DOCX, PPTX)
    file_content = BytesIO(await AsyncPath(file_path).read_bytes())
    file_type = mime.check_file_type(file_content)

    if not is_ocr_required(file_type):
        return process_direct_file(file_content, file_type)

    # OCR processing (PDF, images)
    pages = prepare_pages(file_content, file_type)
    logging.info("Pages: %s", len(pages))
    if not pages:
        return None

    # Process pages with OCR
    text_pages = await process_pages_batch(pages, max_concurrent=10)
    return "\n\n".join([t for t in text_pages if t])


async def process_compressed_archive(
    task: OcrTask, file_content: BytesIO, file_type: str
) -> OcrTask:
    """Process compressed archive and return extracted text."""
    from .services import save_error, save_result

    extracted_archive = archive_utils.extract_archive(file_content, file_type)
    if extracted_archive is None:
        return await save_error(task, "Failed to extract archive")
    temp_dir, extracted_paths = extracted_archive
    if not extracted_paths:
        return await save_error(task, "Failed to extract archive")

    results = await archive_utils.run_directory_files(temp_dir, get_pages)
    total_pages = sum(pages for pages in results if pages)
    quota = await finance.check_quota(task.user_id, total_pages, raise_exception=False)
    if quota < total_pages:
        return await save_error(task, "Insufficient quota")

    await archive_utils.process_directory_files(
        temp_dir, temp_dir / "ocrs", process_file
    )
    zip_buffer = archive_utils.compress_directory_to_zip(temp_dir / "ocrs")
    upload_result = await media.upload_file(zip_buffer)
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Meter usage
    await finance.meter_cost(task.user_id, total_pages)
    return await save_result(task, upload_result)
