"""Unit tests for OCR archive processing."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum


@pytest.mark.unit
class TestProcessCompressedArchive:
    """Tests for process_compressed_archive function."""

    async def test_returns_error_when_extraction_fails(self) -> None:
        """process_compressed_archive should return error when extraction fails."""
        from apps.ocr.archive_services import process_compressed_archive

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.save_report = AsyncMock()

        with patch(
            "apps.ocr.archive_services.archive_utils.extract_archive",
            return_value=(None, []),
        ):
            result = await process_compressed_archive(
                task, BytesIO(b"fake_zip"), "application/zip"
            )

        assert result.task_status == TaskStatusEnum.error

    async def test_returns_error_on_insufficient_quota(self) -> None:
        """process_compressed_archive should return error when quota is insufficient."""
        from apps.ocr.archive_services import process_compressed_archive

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.save_report = AsyncMock()

        mock_temp_dir = MagicMock()
        mock_extracted = [MagicMock()]

        with (
            patch(
                "apps.ocr.archive_services.archive_utils.extract_archive",
                return_value=(mock_temp_dir, mock_extracted),
            ),
            patch(
                "apps.ocr.archive_services.archive_utils.run_directory_files",
                new_callable=AsyncMock,
                return_value=[5],  # 5 pages
            ),
            patch(
                "apps.ocr.archive_services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=0,  # Insufficient quota
            ),
        ):
            result = await process_compressed_archive(
                task, BytesIO(b"fake_zip"), "application/zip"
            )

        assert result.task_status == TaskStatusEnum.error


@pytest.mark.unit
class TestGetPages:
    """Tests for get_pages function."""

    def test_returns_1_for_jpeg_image(self, mock_png_bytes: bytes) -> None:
        """get_pages should return 1 for image files."""
        from pathlib import Path

        from apps.ocr.archive_services import get_pages

        with (
            patch(
                "builtins.open",
                MagicMock(
                    return_value=MagicMock(
                        __enter__=MagicMock(
                            return_value=MagicMock(
                                read=MagicMock(return_value=mock_png_bytes)
                            )
                        ),
                        __exit__=MagicMock(return_value=False),
                    )
                ),
            ),
            patch(
                "apps.ocr.archive_services.mime.check_file_type",
                return_value="image/jpeg",
            ),
        ):
            result = get_pages(Path("test.jpg"))

        assert result == 1

    def test_returns_0_for_unsupported_type(self) -> None:
        """get_pages should return 0 for unsupported file types."""
        from pathlib import Path

        from apps.ocr.archive_services import get_pages

        with (
            patch(
                "builtins.open",
                MagicMock(
                    return_value=MagicMock(
                        __enter__=MagicMock(
                            return_value=MagicMock(read=MagicMock(return_value=b"data"))
                        ),
                        __exit__=MagicMock(return_value=False),
                    )
                ),
            ),
            patch(
                "apps.ocr.archive_services.mime.check_file_type",
                return_value="text/plain",
            ),
        ):
            result = get_pages(Path("test.txt"))

        assert result == 0
