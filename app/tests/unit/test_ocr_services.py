"""Unit tests for OCR services."""

import asyncio
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.ocr import services as services_mod
from apps.ocr.file_processors import (
    can_process_directly,
    is_compressed_file,
    is_docx,
    is_image,
    is_ocr_required,
    is_pdf,
    is_pptx,
)
from apps.ocr.schemas import OcrEngineType
from apps.ocr.services import _resolve_ocr_engine, save_error, save_result


@pytest.mark.unit
class TestFileProcessors:
    """Tests for file type checker functions."""

    def test_is_pdf_returns_true_for_pdf(self) -> None:
        """is_pdf should return True for PDF MIME type."""
        assert is_pdf("application/pdf") is True

    def test_is_pdf_returns_false_for_image(self) -> None:
        """is_pdf should return False for image MIME type."""
        assert is_pdf("image/jpeg") is False

    def test_is_image_returns_true_for_jpeg(self) -> None:
        """is_image should return True for JPEG MIME type."""
        assert is_image("image/jpeg") is True

    def test_is_image_returns_true_for_png(self) -> None:
        """is_image should return True for PNG MIME type."""
        assert is_image("image/png") is True

    def test_is_image_returns_false_for_pdf(self) -> None:
        """is_image should return False for PDF MIME type."""
        assert is_image("application/pdf") is False

    def test_is_docx_returns_true_for_docx(self) -> None:
        """is_docx should return True for DOCX MIME type."""
        assert (
            is_docx(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            is True
        )

    def test_is_pptx_returns_true_for_pptx(self) -> None:
        """is_pptx should return True for PPTX MIME type."""
        assert (
            is_pptx(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
            is True
        )

    def test_is_ocr_required_true_for_pdf(self) -> None:
        """is_ocr_required should return True for PDF."""
        assert is_ocr_required("application/pdf") is True

    def test_is_ocr_required_true_for_image(self) -> None:
        """is_ocr_required should return True for images."""
        assert is_ocr_required("image/jpeg") is True
        assert is_ocr_required("image/png") is True

    def test_is_ocr_required_false_for_docx(self) -> None:
        """is_ocr_required should return False for DOCX (direct extraction)."""
        assert (
            is_ocr_required(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            is False
        )

    def test_is_compressed_file_true_for_zip(self) -> None:
        """is_compressed_file should return True for ZIP."""
        assert is_compressed_file("application/zip") is True

    def test_is_compressed_file_true_for_gzip(self) -> None:
        """is_compressed_file should return True for gzip."""
        assert is_compressed_file("application/gzip") is True

    def test_is_compressed_file_false_for_pdf(self) -> None:
        """is_compressed_file should return False for PDF."""
        assert is_compressed_file("application/pdf") is False

    def test_can_process_directly_true_for_docx(self) -> None:
        """can_process_directly should return True for DOCX."""
        assert (
            can_process_directly(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            is True
        )

    def test_can_process_directly_false_for_pdf(self) -> None:
        """can_process_directly should return False for PDF."""
        assert can_process_directly("application/pdf") is False


@pytest.mark.unit
class TestResolveOcrEngine:
    """Tests for _resolve_ocr_engine function."""

    def test_returns_llm_by_default(self) -> None:
        """_resolve_ocr_engine should return llm when no engine specified."""
        task = MagicMock()
        task.ocr_engine = None

        with patch("apps.ocr.services.Settings") as mock_settings:
            mock_settings.ocr_engine = "llm"
            result = _resolve_ocr_engine(task)

        assert result == OcrEngineType.llm

    def test_returns_paddle_for_paddle_alias(self) -> None:
        """_resolve_ocr_engine should resolve 'paddle' alias to paddleocr_vl_1_5."""
        task = MagicMock()
        task.ocr_engine = "paddle"

        with patch("apps.ocr.services.Settings") as mock_settings:
            mock_settings.ocr_engine = "llm"
            result = _resolve_ocr_engine(task)

        assert result == OcrEngineType("paddleocr_vl_1_5")

    def test_task_engine_overrides_settings(self) -> None:
        """_resolve_ocr_engine should use task engine over settings."""
        task = MagicMock()
        task.ocr_engine = "llm"

        with patch("apps.ocr.services.Settings") as mock_settings:
            mock_settings.ocr_engine = "paddle"
            result = _resolve_ocr_engine(task)

        assert result == OcrEngineType.llm

    def test_returns_document_intelligence_for_di_alias(self) -> None:
        """_resolve_ocr_engine should resolve 'di' alias to document_intelligence."""
        task = MagicMock()
        task.ocr_engine = "di"

        with patch("apps.ocr.services.Settings") as mock_settings:
            mock_settings.ocr_engine = "pipeline"
            result = _resolve_ocr_engine(task)

        assert result == OcrEngineType.document_intelligence


@pytest.mark.unit
class TestSaveError:
    """Tests for save_error function."""

    async def test_sets_error_status(self) -> None:
        """save_error should set task status to error."""
        task = MagicMock()
        task.save_report = AsyncMock()

        await save_error(task, "Something went wrong")

        assert task.task_status == TaskStatusEnum.error
        task.save_report.assert_called_once_with("Something went wrong")

    async def test_returns_task(self) -> None:
        """save_error should return the task."""
        task = MagicMock()
        task.save_report = AsyncMock()

        result = await save_error(task, "error message")

        assert result is task


@pytest.mark.unit
class TestSaveResult:
    """Tests for save_result function."""

    async def test_sets_completed_status(self) -> None:
        """save_result should set task status to completed."""
        task = MagicMock()
        task.save_report = AsyncMock()

        await save_result(task, "Extracted text")

        assert task.task_status == TaskStatusEnum.completed

    async def test_normalizes_and_saves_result(self) -> None:
        """save_result should normalize text and save it."""
        task = MagicMock()
        task.save_report = AsyncMock()

        await save_result(task, "  Text with spaces  ")

        assert task.result == "Text with spaces"

    async def test_saves_usage_info(self) -> None:
        """save_result should save usage amount and ID."""
        task = MagicMock()
        task.save_report = AsyncMock()

        await save_result(task, "text", usage_amount=5.0, usage_id="usage_123")

        assert task.usage_amount == pytest.approx(5.0)
        assert task.usage_id == "usage_123"


@pytest.mark.unit
class TestProcessOcr:
    """Tests for process_ocr function."""

    async def test_processes_image_with_llm_engine(self) -> None:
        """process_ocr should process images using LLM OCR engine."""
        from apps.ocr.services import process_ocr

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.ocr_engine = None
        task.file_content = AsyncMock(return_value=BytesIO(b"fake_image_data"))
        task.save_report = AsyncMock()

        with (
            patch("apps.ocr.services.mime.check_file_type", return_value="image/jpeg"),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=True),
            patch(
                "apps.ocr.services.prepare_pages",
                return_value=["page1_data"],
            ),
            patch(
                "apps.ocr.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=100,
            ),
            patch(
                "apps.ocr.services.process_pages_batch",
                new_callable=AsyncMock,
                return_value=["Extracted text from image"],
            ),
            patch(
                "apps.ocr.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=MagicMock(amount=1.0, uid="usage_123"),
            ),
            patch("apps.ocr.services.Settings") as mock_settings,
        ):
            mock_settings.ocr_engine = "llm"
            result = await process_ocr(task)

        assert result.task_status == TaskStatusEnum.completed

    async def test_returns_error_on_insufficient_quota(self) -> None:
        """process_ocr should return error when quota is insufficient."""
        from apps.ocr.services import process_ocr

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.ocr_engine = None
        task.file_content = AsyncMock(return_value=BytesIO(b"fake_image_data"))
        task.save_report = AsyncMock()

        with (
            patch("apps.ocr.services.mime.check_file_type", return_value="image/jpeg"),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=True),
            patch("apps.ocr.services.prepare_pages", return_value=["page1"]),
            patch(
                "apps.ocr.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=0,  # Insufficient quota
            ),
            patch("apps.ocr.services.Settings") as mock_settings,
        ):
            mock_settings.ocr_engine = "llm"
            result = await process_ocr(task)

        assert result.task_status == TaskStatusEnum.error

    async def test_processes_docx_directly(self) -> None:
        """process_ocr should process DOCX files without OCR."""
        from apps.ocr.services import process_ocr

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.ocr_engine = None
        task.file_content = AsyncMock(return_value=BytesIO(b"fake_docx_data"))
        task.save_report = AsyncMock()

        docx_mime = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        with (
            patch("apps.ocr.services.mime.check_file_type", return_value=docx_mime),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=False),
            patch(
                "apps.ocr.services.process_direct_file",
                return_value="Extracted DOCX text",
            ),
            patch("apps.ocr.services.Settings") as mock_settings,
        ):
            mock_settings.ocr_engine = "llm"
            result = await process_ocr(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result == "Extracted DOCX text"


@pytest.mark.unit
class TestOcrQuotaAndErrorHandling:
    """Tests for OCR quota checking, metering, and error handling."""

    async def test_quota_checked_before_processing(self) -> None:
        """process_ocr should check quota before processing OCR tasks."""
        from apps.ocr.services import process_ocr

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.workspace_id = None
        task.ocr_engine = None
        task.file_content = AsyncMock(return_value=BytesIO(b"fake_image_data"))
        task.save_report = AsyncMock()

        with (
            patch("apps.ocr.services.mime.check_file_type", return_value="image/jpeg"),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=True),
            patch("apps.ocr.services.prepare_pages", return_value=["page1", "page2"]),
            patch(
                "apps.ocr.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=100,
            ) as mock_check_quota,
            patch(
                "apps.ocr.services.process_pages_batch",
                new_callable=AsyncMock,
                return_value=["Text 1", "Text 2"],
            ),
            patch(
                "apps.ocr.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=MagicMock(amount=2.0, uid="usage_123"),
            ),
            patch("apps.ocr.services.Settings") as mock_settings,
        ):
            mock_settings.ocr_engine = "llm"
            await process_ocr(task)

        # Verify quota was checked with correct parameters
        mock_check_quota.assert_called_once_with(
            "user_123", 2, raise_exception=False, workspace_id=None
        )

    async def test_usage_metered_after_processing(self) -> None:
        """process_ocr should meter usage after successful OCR processing."""
        from apps.ocr.services import process_ocr

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.workspace_id = None
        task.ocr_engine = None
        task.file_content = AsyncMock(return_value=BytesIO(b"fake_image_data"))
        task.save_report = AsyncMock()

        with (
            patch("apps.ocr.services.mime.check_file_type", return_value="image/png"),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=True),
            patch(
                "apps.ocr.services.prepare_pages",
                return_value=["page1", "page2", "page3"],
            ),
            patch(
                "apps.ocr.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=100,
            ),
            patch(
                "apps.ocr.services.process_pages_batch",
                new_callable=AsyncMock,
                return_value=["Text 1", "Text 2", "Text 3"],
            ),
            patch(
                "apps.ocr.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=MagicMock(amount=3.6, uid="usage_456"),
            ) as mock_meter_cost,
            patch("apps.ocr.services.Settings") as mock_settings,
        ):
            mock_settings.ocr_engine = "llm"
            result = await process_ocr(task)

        # Verify metering was called with correct parameters
        mock_meter_cost.assert_called_once_with(
            "user_123",
            pytest.approx(3.6),
            meta_data={
                "service": "ocr",
                "engine": "llm",
                "pages": 3,
                "task_uid": "task_123",
            },
            workspace_id=None,
        )

        # Verify usage info was saved to task
        assert result.usage_amount == pytest.approx(3.6)
        assert result.usage_id == "usage_456"

    async def test_error_on_unsupported_file_type(self) -> None:
        """process_ocr should return error for files that cannot be prepared."""
        from apps.ocr.services import process_ocr

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.ocr_engine = None
        task.file_content = AsyncMock(return_value=BytesIO(b"fake_data"))
        task.save_report = AsyncMock()

        with (
            patch(
                "apps.ocr.services.mime.check_file_type",
                return_value="application/octet-stream",
            ),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=True),
            patch(
                "apps.ocr.services.prepare_pages", return_value=[]
            ),  # Empty pages = unsupported
            patch("apps.ocr.services.Settings") as mock_settings,
        ):
            mock_settings.ocr_engine = "llm"
            result = await process_ocr(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_called_once()
        # Verify error message mentions file type
        error_message = task.save_report.call_args[0][0]
        assert "application/octet-stream" in error_message

    async def test_insufficient_quota_error_message(self) -> None:
        """process_ocr should save 'insufficient_quota' error message."""
        from apps.ocr.services import process_ocr

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.ocr_engine = None
        task.file_content = AsyncMock(return_value=BytesIO(b"fake_image_data"))
        task.save_report = AsyncMock()

        with (
            patch("apps.ocr.services.mime.check_file_type", return_value="image/jpeg"),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=True),
            patch(
                "apps.ocr.services.prepare_pages",
                return_value=["page1", "page2", "page3"],
            ),
            patch(
                "apps.ocr.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=2,  # Quota less than required pages
            ),
            patch("apps.ocr.services.Settings") as mock_settings,
        ):
            mock_settings.ocr_engine = "llm"
            result = await process_ocr(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_called_once_with("insufficient_quota:2:3")

    async def test_no_quota_check_for_direct_processing(self) -> None:
        """process_ocr should not check quota for DOCX/PPTX direct processing."""
        from apps.ocr.services import process_ocr

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.ocr_engine = None
        task.file_content = AsyncMock(return_value=BytesIO(b"fake_docx_data"))
        task.save_report = AsyncMock()

        docx_mime = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        with (
            patch("apps.ocr.services.mime.check_file_type", return_value=docx_mime),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=False),
            patch(
                "apps.ocr.services.process_direct_file",
                return_value="Extracted DOCX text",
            ),
            patch(
                "apps.ocr.services.finance.check_quota", new_callable=AsyncMock
            ) as mock_check_quota,
            patch("apps.ocr.services.Settings") as mock_settings,
        ):
            mock_settings.ocr_engine = "llm"
            result = await process_ocr(task)

        # Verify quota was NOT checked for direct processing
        mock_check_quota.assert_not_called()
        assert result.task_status == TaskStatusEnum.completed

    async def test_error_handling_with_exception(self) -> None:
        """process_ocr should handle unexpected exceptions gracefully."""
        from apps.ocr.services import process_ocr

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.ocr_engine = None
        task.file_content = AsyncMock(return_value=BytesIO(b"fake_image_data"))
        task.save_report = AsyncMock()

        with (
            patch(
                "apps.ocr.services.mime.check_file_type",
                side_effect=Exception("Unexpected error"),
            ),
            patch("apps.ocr.services.Settings") as mock_settings,
        ):
            mock_settings.ocr_engine = "llm"
            result = await process_ocr(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_called_once_with(
            "OCR processing failed: Unexpected error"
        )


@pytest.mark.unit
class TestResumeStuckOcrTasks:
    """
    Startup crash-recovery reconciler.

    A task still "processing" when the process starts up was never marked
    completed/error by whatever process handled it before -- since OCR
    runs as an in-process background task, that only happens if the
    previous process died mid-job.
    """

    async def test_reprocesses_every_task_still_marked_processing(self) -> None:
        from apps.ocr.services import resume_stuck_ocr_tasks

        stuck_a, stuck_b = MagicMock(uid="a"), MagicMock(uid="b")
        find_result = MagicMock()
        find_result.to_list = AsyncMock(return_value=[stuck_a, stuck_b])

        with (
            patch(
                "apps.ocr.services.OcrTask.find", return_value=find_result
            ) as mock_find,
            patch(
                "apps.ocr.services.process_ocr", new_callable=AsyncMock
            ) as mock_process,
        ):
            await resume_stuck_ocr_tasks()
            # Let the fire-and-forget asyncio.create_task callbacks run.
            for task in list(services_mod._background_tasks):
                await task

        mock_find.assert_called_once_with({"task_status": TaskStatusEnum.processing.value})
        assert mock_process.await_args_list == [
            ((stuck_a,),),
            ((stuck_b,),),
        ]

    async def test_no_stuck_tasks_schedules_nothing(self) -> None:
        from apps.ocr.services import resume_stuck_ocr_tasks

        find_result = MagicMock()
        find_result.to_list = AsyncMock(return_value=[])

        with (
            patch("apps.ocr.services.OcrTask.find", return_value=find_result),
            patch(
                "apps.ocr.services.process_ocr", new_callable=AsyncMock
            ) as mock_process,
        ):
            await resume_stuck_ocr_tasks()

        mock_process.assert_not_called()


def _fake_task() -> SimpleNamespace:
    return SimpleNamespace(
        uid="t1",
        user_id="u1",
        workspace_id=None,
        save_report=AsyncMock(),
        result=None,
        task_status=None,
        usage_amount=None,
        usage_id=None,
        provider_meta=None,
    )


def _fake_di_result(tmp_path: Path, assets: list) -> SimpleNamespace:
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    return SimpleNamespace(
        markdown="di-md",
        assets=assets,
        docx_bytes=b"PK",
        output_dir=out_dir,
        stats={},
    )


@pytest.mark.unit
class TestDocumentIntelligenceAssetUploadConcurrency:
    """
    Document Intelligence asset uploads must run with bounded concurrency,
    not one-at-a-time -- see ``_process_with_document_intelligence``.
    """

    async def _run_with_assets(
        self, tmp_path: Path, assets: list, upload_file_mock: AsyncMock
    ):
        from apps.ocr import services as svc
        from apps.ocr.schemas import OcrEngineType

        di = MagicMock()
        di.process = AsyncMock(return_value=_fake_di_result(tmp_path, assets))
        di.cleanup = MagicMock()
        di.output_dir = tmp_path / "di"
        usage = SimpleNamespace(amount=1.0, uid="usage1")

        with (
            patch("apps.ocr.pipeline.renderer.count_pdf_bytes", return_value=1),
            patch("apps.ocr.services.finance.check_quota", AsyncMock(return_value=10)),
            patch(
                "apps.ocr.document_intelligence.DocumentIntelligencePipeline",
                return_value=di,
            ),
            patch("apps.ocr.document_intelligence.summarize_stats", return_value={}),
            patch(
                "apps.ocr.document_intelligence.renderers.markdown.rewrite_asset_links",
                side_effect=lambda markdown, url_map: markdown,
            ) as rewrite_mock,
            patch("utils.integrations.media.upload_file", upload_file_mock),
            patch("apps.ocr.services.finance.estimate_ocr_cost", return_value=1.0),
            patch(
                "apps.ocr.services.finance.meter_cost", AsyncMock(return_value=usage)
            ),
            patch("apps.ocr.services.texttools.normalize_text", return_value="di-md"),
        ):
            out = await svc._process_with_document_intelligence(
                _fake_task(),
                BytesIO(b"%PDF"),
                "application/pdf",
                OcrEngineType.document_intelligence,
            )
        return out, rewrite_mock

    @pytest.mark.asyncio
    async def test_assets_upload_concurrently_not_serially(
        self, tmp_path: Path
    ) -> None:
        """
        N asset uploads that each take ``delay`` seconds should finish in
        roughly one ``delay``-sized wave (plus the docx upload's own
        ``delay``), not ``N * delay`` -- proving the upload loop is no
        longer a plain sequential ``for asset in result.assets: await ...``.
        """
        n_assets = 6
        delay = 0.12
        assets = []
        for i in range(n_assets):
            path = tmp_path / f"asset{i}.png"
            path.write_bytes(b"img")
            assets.append(
                SimpleNamespace(path=str(path), rel_path=f"assets/asset{i}.png")
            )

        async def slow_upload_file(buf, *, user_id, workspace_id=None):
            await asyncio.sleep(delay)
            return "https://media/ok"

        start = time.monotonic()
        out, _ = await self._run_with_assets(
            tmp_path, assets, AsyncMock(side_effect=slow_upload_file)
        )
        elapsed = time.monotonic() - start

        assert out.result == "di-md"
        # Fully serial would be (n_assets + 1 docx) * delay. Concurrent
        # uploads plus one sequential docx upload should land near 2*delay.
        serial_time = (n_assets + 1) * delay
        assert elapsed < serial_time * 0.6, (
            f"elapsed={elapsed:.3f}s not much faster than serial={serial_time:.3f}s "
            "-- asset uploads look serialized"
        )

    @pytest.mark.asyncio
    async def test_one_failed_asset_upload_does_not_abort_others(
        self, tmp_path: Path
    ) -> None:
        """A single asset failing to upload must not drop the others."""
        good_a = tmp_path / "good_a.png"
        good_a.write_bytes(b"img")
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"img")
        good_b = tmp_path / "good_b.png"
        good_b.write_bytes(b"img")

        assets = [
            SimpleNamespace(path=str(good_a), rel_path="assets/good_a.png"),
            SimpleNamespace(path=str(bad), rel_path="assets/bad.png"),
            SimpleNamespace(path=str(good_b), rel_path="assets/good_b.png"),
        ]

        # All three assets have identical content, so the only reliable way
        # to make exactly one fail deterministically (regardless of
        # completion order under concurrency) is by call count.
        call_count = 0

        async def _side_effect(buf, *, user_id, workspace_id=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                error = RuntimeError("upload failed")
                raise error
            return f"https://media/ok-{call_count}"

        upload_mock = AsyncMock(side_effect=_side_effect)

        out, rewrite_mock = await self._run_with_assets(tmp_path, assets, upload_mock)

        assert out.result == "di-md"
        rewrite_mock.assert_called_once()
        (_markdown, url_map), _kwargs = rewrite_mock.call_args
        # Exactly one of the three assets failed; the other two must still
        # have made it into the url_map (per-asset failure isolation, now
        # under concurrency).
        assert len(url_map) == 2
