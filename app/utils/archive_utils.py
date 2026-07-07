"""Simple archive utilities - extract to temp directory, compress from directory."""

import asyncio
import shutil
from collections.abc import Awaitable, Callable
from io import BytesIO
from pathlib import Path

from anyio import Path as AsyncPath

COMPRESSED_EXTS = {
    "application/zip",
    "application/x-zip-compressed",
    "application/x-bzip2",
    "application/x-bz2",
    "application/zstd",
    "application/gzip",
    "application/x-gzip",
    "application/x-tar",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
}


def _extract_zip(file_content: BytesIO, temp_dir: Path) -> list[Path]:
    """Extract ZIP file to temp directory."""
    import zipfile

    extracted_paths = []
    with zipfile.ZipFile(file_content) as zip_file:
        for file_info in zip_file.filelist:
            if not file_info.is_dir():
                extracted_path = temp_dir / file_info.filename
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                with open(extracted_path, "wb") as f:
                    f.write(zip_file.read(file_info.filename))
                extracted_paths.append(extracted_path)
    return extracted_paths


def _extract_tar(file_content: BytesIO, temp_dir: Path) -> list[Path]:
    """Extract TAR file to temp directory."""
    import tarfile

    extracted_paths = []
    with tarfile.open(fileobj=file_content, mode="r") as tar_file:
        for member in tar_file.getmembers():
            if member.isfile():
                extracted_path = temp_dir / member.name
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                with open(extracted_path, "wb") as f:
                    f.write(tar_file.extractfile(member).read())
                extracted_paths.append(extracted_path)
    return extracted_paths


def _extract_single_file(
    file_content: BytesIO, temp_dir: Path, decompress_func: Callable[[bytes], bytes]
) -> list[Path]:
    """Extract single compressed file to temp directory."""
    extracted_paths = []
    decompressed = decompress_func(file_content.read())
    extracted_path = temp_dir / "extracted_file"
    extracted_path.parent.mkdir(parents=True, exist_ok=True)
    with open(extracted_path, "wb") as f:
        f.write(decompressed)
    extracted_paths.append(temp_dir / "extracted_file")
    return extracted_paths


def extract_archive(
    file_content: BytesIO, file_type: str
) -> tuple[Path, list[Path]] | None:
    """Extract any supported archive to a temporary directory."""
    import bz2
    import gzip
    import tempfile

    import zstandard as zstd

    temp_dir = Path(tempfile.mkdtemp())

    try:
        if file_type in {"application/zip", "application/x-zip-compressed"}:
            extracted_paths = _extract_zip(file_content, temp_dir)
        elif file_type == "application/x-tar":
            extracted_paths = _extract_tar(file_content, temp_dir)
        elif file_type in {"application/gzip", "application/x-gzip"}:
            extracted_paths = _extract_single_file(
                file_content, temp_dir, gzip.decompress
            )
        elif file_type in {"application/x-bzip2", "application/x-bz2"}:
            extracted_paths = _extract_single_file(
                file_content, temp_dir, bz2.decompress
            )
        elif file_type == "application/zstd":
            extracted_paths = _extract_single_file(
                file_content, temp_dir, zstd.ZstdDecompressor().decompress
            )
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    return temp_dir, extracted_paths


def compress_directory_to_zip(directory: Path) -> BytesIO:
    """Compress a directory to ZIP and return as BytesIO."""
    import zipfile

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                zip_file.write(file_path, file_path.relative_to(directory))
    zip_buffer.seek(0)
    return zip_buffer


async def run_directory_files(
    input_dir: Path,
    file_processor: Callable[[Path], object | Awaitable[object] | None],
) -> list[Path]:
    """
    Process all files in a directory using the provided processor.

    Args:
        input_dir: Directory containing files to process.
        file_processor: Callable that processes each file path.

    Returns:
        List of results from processing each file.
    """
    file_paths = [
        file_path
        async for file_path in AsyncPath(input_dir).rglob("*")
        if await file_path.is_file()
    ]

    semaphore = asyncio.Semaphore(10)

    async def process_path(file_path: AsyncPath) -> str | None:
        async with semaphore:
            if asyncio.iscoroutinefunction(file_processor):
                return await file_processor(file_path)
            else:
                return file_processor(file_path)

    results = await asyncio.gather(*(process_path(p) for p in file_paths))
    return results


async def process_directory_files(
    input_dir: Path,
    output_dir: Path,
    file_processor: Callable[[Path], str | Awaitable[str] | None],
) -> list[Path]:
    """
    Process all files in a directory and write results as .txt files.

    Args:
        input_dir: Directory containing files to process.
        output_dir: Directory where result files will be written.
        file_processor: Callable that processes each file and returns text.

    Returns:
        List of paths to the written output files.
    """

    file_paths = [
        file_path
        async for file_path in AsyncPath(input_dir).rglob("*")
        if await file_path.is_file()
    ]

    semaphore = asyncio.Semaphore(10)

    async def process_path(file_path: AsyncPath) -> str | None:
        async with semaphore:
            if asyncio.iscoroutinefunction(file_processor):
                text = await file_processor(file_path)
            else:
                text = file_processor(file_path)

            if text:
                # Preserve directory structure
                relative_path = file_path.relative_to(input_dir)
                output_file = AsyncPath(output_dir / relative_path.with_suffix(".md"))
                await output_file.parent.mkdir(parents=True, exist_ok=True)
                await output_file.write_text(text, encoding="utf-8")
                return Path(output_file)

    results = await asyncio.gather(*(process_path(p) for p in file_paths))

    return results
