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
        captured_hints: list[str] = []

        async def ocr_fn(image: object, layout_hint: str = "") -> str:
            captured_hints.append(layout_hint)
            return "extracted page text"

        pipeline = DocumentPipeline(
            enable_preprocessing=False,
            enable_layout=True,
            enable_normalization=False,
            pipeline_ocr_fn=ocr_fn,
        )
        pipeline.layout_detector.detect = MagicMock(return_value=elements)

        result = await pipeline.process_image(Image.new("RGB", (100, 60)))

        assert result == "extracted page text"
        assert len(captured_hints) == 1
        assert "title" in captured_hints[0]
        assert "paragraph" in captured_hints[0]


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
            result = await ocr_to_text(image_bytes, layout_hint="[1] (paragraph) y=0–50")

        assert result == "متن"
        assert complete.await_args.args[0]["model"] == "google/gemini-3.1-flash-lite"
