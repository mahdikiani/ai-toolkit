"""Unit tests for apps.ocr.document_intelligence.elements."""

from unittest.mock import AsyncMock

import pytest
from PIL import Image

from apps.ocr.document_intelligence.elements import (
    ElementProcessor,
    _split_caption_description,
)
from apps.ocr.document_intelligence.layout import LayoutElement, LayoutType


@pytest.mark.document_intelligence
class TestSplitCaptionDescription:
    r"""
    A VLM's free-text response must never leak into the caption when it
    doesn't follow the requested "caption: ...\ndescription: ..." format
    -- caption renders as real visible text, description becomes image
    alt text, so an unparsed response must fall back to description-only.
    """

    def test_well_formed_response_splits_both_parts(self) -> None:
        text = "caption: Figure 1\ndescription: A scatter plot of two groups."
        caption, description = _split_caption_description(text)
        assert caption == "Figure 1"
        assert description == "A scatter plot of two groups."

    def test_capitalized_labels_still_split(self) -> None:
        """
        Regression: a previous version matched "description:" against
        a lowercased copy but sliced the original-cased text with it, so
        an actual "Description:" response silently failed to split."""
        text = "Caption: Figure 1\nDescription: A scatter plot of two groups."
        caption, description = _split_caption_description(text)
        assert caption == "Figure 1"
        assert description == "A scatter plot of two groups."

    def test_unparseable_response_becomes_description_only(self) -> None:
        """
        Regression: this exact free-text shape used to make caption
        AND description both equal the full text, which visibly leaked
        the whole alt-text-style blob into the document body via the
        (legitimately-rendered) caption field."""
        text = "A scatter plot showing two distinct groups of data points."
        caption, description = _split_caption_description(text)
        assert caption == ""
        assert description == text

    def test_caption_without_description_becomes_description_only(self) -> None:
        text = "caption: only a caption label, no description marker"
        caption, description = _split_caption_description(text)
        assert caption == ""
        assert description == text

    def test_description_before_caption_becomes_description_only(self) -> None:
        """
        Out-of-order labels aren't a format this parses -- treat as
        unparseable rather than guessing."""
        text = "description: the real description\ncaption: a label"
        caption, description = _split_caption_description(text)
        assert caption == ""
        assert description == text

    def test_overlong_caption_folds_into_description(self) -> None:
        """
        Regression: the VLM labeled a full sentence as "caption:"
        (technically valid format, but not a real caption) -- it must
        still not render as visible text just because it followed the
        two-field format the prompt asked for."""
        text = (
            "caption: A scatter plot showing two distinct classes of data "
            "points separated by a horizontal line.\n"
            "description: A 2D coordinate system with two clusters."
        )
        caption, description = _split_caption_description(text)
        assert caption == ""
        assert "scatter plot" in description
        assert "coordinate system" in description

    def test_short_caption_with_colon_still_counts_as_short(self) -> None:
        text = "caption: شکل ۱: داده‌های سوال\ndescription: a chart."
        caption, description = _split_caption_description(text)
        assert caption == "شکل ۱: داده‌های سوال"
        assert description == "a chart."


def _elem(elem_type: LayoutType) -> LayoutElement:
    return LayoutElement(
        id="p0001_e0001",
        page_id="p0001",
        page_number=1,
        type=elem_type,
        bbox=(0, 0, 100, 100),
        padded_bbox=(0, 0, 100, 100),
        confidence=0.9,
    )


@pytest.mark.document_intelligence
class TestDigitNormalization:
    """
    Regression: the VLM inconsistently emits Arabic-Indic digits
    instead of Persian ones, most often for 4/5/6 whose glyphs
    genuinely differ between the two scripts -- every handler that
    produces reader-visible text must normalize digits before they
    reach the rendered document.
    """

    @pytest.fixture
    def crop(self) -> Image.Image:
        return Image.new("RGB", (10, 10))

    async def test_process_text_normalizes_digits(self, crop: Image.Image) -> None:
        processor = ElementProcessor(vlm_model="test-model")
        processor._vlm_call = AsyncMock(return_value="سال ٤٥٦")

        result = await processor._process_text(crop, _elem(LayoutType.paragraph))

        assert result.text == "سال ۴۵۶"

    async def test_process_table_normalizes_digits_without_touching_markup(
        self, crop: Image.Image
    ) -> None:
        processor = ElementProcessor(vlm_model="test-model")
        processor._vlm_call = AsyncMock(
            return_value="<table><tr><td>٤</td><td>٦</td></tr></table>"
        )

        result = await processor._process_table(crop, _elem(LayoutType.table))

        assert result.html == "<table><tr><td>۴</td><td>۶</td></tr></table>"
        assert result.text == result.html

    async def test_process_figure_normalizes_digits_in_caption_and_description(
        self, crop: Image.Image
    ) -> None:
        processor = ElementProcessor(vlm_model="test-model")
        processor._vlm_call = AsyncMock(
            return_value="caption: شکل ٤\ndescription: نمودار سال ٥٦."
        )

        result = await processor._process_figure(crop, _elem(LayoutType.figure))

        assert result.caption == "شکل ۴"
        assert result.description == "نمودار سال ۵۶."

    async def test_process_chart_normalizes_title_and_description_only(
        self, crop: Image.Image
    ) -> None:
        processor = ElementProcessor(vlm_model="test-model")
        processor._vlm_call = AsyncMock(
            return_value=(
                '{"chart_type": "bar", "title": "نمودار ٤",'
                ' "description": "داده سال ٥٦.",'
                ' "data": [{"label": "٤", "value": 4}]}'
            )
        )

        result = await processor._process_chart(crop, _elem(LayoutType.chart))

        assert result.caption == "نمودار ۴"
        assert result.description == "داده سال ۵۶."
        # chart_data's own numeric payload is left untouched so exact
        # values (used for e.g. plotting) aren't altered.
        assert result.chart_data["data"][0]["label"] == "٤"
