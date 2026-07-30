"""Unit tests for layout-aware document OCR."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from apps.ocr.ocr_services import ocr_to_text
from apps.ocr.pipeline.engine import DocumentPipeline
from apps.ocr.pipeline.layout_detector import ElementType, LayoutBox
from apps.ocr.pipeline.reading_order import ReadingOrderResolver


@pytest.mark.unit
class TestDocumentPipeline:
    """Test document segmentation and block reconstruction."""

    async def test_processes_full_page_with_layout_hint(self) -> None:
        """Full page is sent to OCR; layout is passed as text annotation."""
        elements = [
            LayoutBox("first", 1, ElementType.title, 0, 0, 100, 20),
            LayoutBox("second", 1, ElementType.paragraph, 0, 25, 100, 50),
        ]
        captured_hints: list[object] = []

        async def ocr_fn(image: object, element: object | None = None) -> str:
            captured_hints.append(element)
            return "extracted page text"

        pipeline = DocumentPipeline(
            enable_preprocessing=False,
            enable_layout=True,
            enable_normalization=False,
            pipeline_ocr_fn=ocr_fn,
        )
        pipeline.layout_detector.detect = MagicMock(return_value=elements)

        result = await pipeline.process_image(Image.new("RGB", (100, 60)))

        assert "# extracted page text" in result
        assert "extracted page text" in result
        assert len(captured_hints) == 2
        assert captured_hints[0] is not None
        assert captured_hints[1] is not None

    async def test_on_page_done_called_after_every_page(self) -> None:
        """Progress callback fires with (page_num, total) after each page."""
        pipeline = DocumentPipeline(
            enable_preprocessing=False,
            enable_layout=False,
            enable_normalization=False,
            pipeline_ocr_fn=AsyncMock(return_value="text"),
        )
        images = [Image.new("RGB", (50, 50)) for _ in range(3)]
        progress_calls: list[tuple[int, int]] = []

        async def on_page_done(page_num: int, total: int) -> None:
            progress_calls.append((page_num, total))

        await pipeline._process_images(images, on_page_done=on_page_done)

        assert progress_calls == [(1, 3), (2, 3), (3, 3)]

    async def test_one_failed_page_does_not_abort_the_document(self) -> None:
        """A single page's unexpected exception is isolated, not fatal."""
        pipeline = DocumentPipeline(
            enable_preprocessing=False,
            enable_layout=False,
            enable_normalization=False,
            pipeline_ocr_fn=AsyncMock(return_value="page text"),
        )
        images = [Image.new("RGB", (50, 50)) for _ in range(3)]
        original_process_page = pipeline._process_page

        async def flaky_process_page(image: object, page_number: int) -> object:
            if page_number == 2:
                error = RuntimeError("corrupt page")
                raise error
            return await original_process_page(image, page_number)

        pipeline._process_page = flaky_process_page

        result = await pipeline._process_images(images)

        assert "page text" in result
        assert "صفحه 2" in result

    async def test_progress_callback_failure_does_not_abort_the_document(self) -> None:
        """A broken progress callback must not take down the whole job."""
        pipeline = DocumentPipeline(
            enable_preprocessing=False,
            enable_layout=False,
            enable_normalization=False,
            pipeline_ocr_fn=AsyncMock(return_value="text"),
        )
        images = [Image.new("RGB", (50, 50))]

        async def broken_on_page_done(page_num: int, total: int) -> None:
            error = RuntimeError("webhook down")
            raise error

        result = await pipeline._process_images(images, on_page_done=broken_on_page_done)

        assert "text" in result

    async def test_checkpointed_page_is_reused_instead_of_reprocessed(self) -> None:
        """A page found in Redis checkpoints must not be run through OCR again."""
        ocr_fn = AsyncMock(return_value="fresh text")
        pipeline = DocumentPipeline(
            enable_preprocessing=False,
            enable_layout=False,
            enable_normalization=False,
            pipeline_ocr_fn=ocr_fn,
        )
        images = [Image.new("RGB", (50, 50)) for _ in range(2)]
        checkpoints = {
            1: {"text": "cached page one", "assets": []},
        }

        with patch(
            "apps.ocr.pipeline.engine.checkpoint_store.load_pages",
            AsyncMock(return_value=checkpoints),
        ):
            result = await pipeline._process_images(images, task_uid="task-1")

        assert "cached page one" in result
        assert "fresh text" in result
        # Only page 2 should have actually gone through OCR.
        assert ocr_fn.await_count == 1


@pytest.mark.unit
class TestReadingOrder:
    """Test right-to-left reading order behavior."""

    def test_orders_same_row_right_to_left(self) -> None:
        """Persian blocks on the same row start from the rightmost block."""
        left = LayoutBox("left", 1, ElementType.paragraph, 10, 10, 40, 40)
        right = LayoutBox("right", 1, ElementType.paragraph, 60, 10, 90, 40)
        elements = [left, right]

        ReadingOrderResolver().resolve(elements)

        assert [element.element_id for element in elements] == ["right", "left"]


@pytest.mark.unit
class TestVisionOcr:
    """Test the configured vision model request."""

    async def test_uses_configured_ocr_vlm_model(self) -> None:
        """OCR block extraction uses the dedicated current Gemini model."""
        image = Image.new("RGB", (16, 16), "white")
        image_bytes = BytesIO()
        image.save(image_bytes, format="PNG")
        response = {"choices": [{"message": {"content": "متن"}}]}

        with patch(
            "apps.ocr.ocr_services.complete_chat_json",
            new_callable=AsyncMock,
            return_value=response,
        ) as complete:
            result = await ocr_to_text(image_bytes, block_type="paragraph")

        assert result == "متن"
        assert complete.await_args.args[0]["model"] == "google/gemini-3.1-flash-lite"
