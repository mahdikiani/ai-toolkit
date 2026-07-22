"""Unit tests for the Document Intelligence Reading Order resolver."""

import pytest

from apps.ocr.document_intelligence.layout import LayoutElement, LayoutType
from apps.ocr.document_intelligence.reading_order import ReadingOrderResolver


def _elem(elem_id: str, x1: float, y1: float, x2: float, y2: float) -> LayoutElement:
    return LayoutElement(
        id=elem_id,
        page_id="p0001",
        page_number=1,
        type=LayoutType.paragraph,
        bbox=(x1, y1, x2, y2),
        padded_bbox=(x1, y1, x2, y2),
        confidence=0.9,
    )


@pytest.mark.unit
class TestDetectIsRtl:
    """Tests for auto language/direction detection."""

    def test_defaults_to_rtl_when_no_text(self) -> None:
        """No extracted text yet -> assume RTL (Persian/Arabic target documents)."""
        assert ReadingOrderResolver.detect_is_rtl([], None) is True
        assert ReadingOrderResolver.detect_is_rtl([], {}) is True

    def test_detects_rtl_for_persian_text(self) -> None:
        elem = _elem("e1", 0, 0, 10, 10)
        assert ReadingOrderResolver.detect_is_rtl([elem], {"e1": "سلام دنیا، این یک متن فارسی است"}) is True

    def test_detects_ltr_for_english_text(self) -> None:
        elem = _elem("e1", 0, 0, 10, 10)
        assert ReadingOrderResolver.detect_is_rtl([elem], {"e1": "hello world this is english text"}) is False


@pytest.mark.unit
class TestResolveFallbackOrdering:
    """Tests for the no-column fallback ordering path (< 3 candidate elements)."""

    def test_rtl_sorts_top_to_bottom_then_right_to_left(self) -> None:
        left = _elem("left", 10, 10, 40, 40)
        right = _elem("right", 60, 10, 90, 40)

        ordered = ReadingOrderResolver().resolve([left, right], page_width=100, is_rtl=True)

        assert [e.id for e in ordered] == ["right", "left"]

    def test_ltr_sorts_top_to_bottom_then_left_to_right(self) -> None:
        left = _elem("left", 10, 10, 40, 40)
        right = _elem("right", 60, 10, 90, 40)

        ordered = ReadingOrderResolver().resolve([left, right], page_width=100, is_rtl=False)

        assert [e.id for e in ordered] == ["left", "right"]

    def test_auto_detects_direction_from_texts_when_is_rtl_omitted(self) -> None:
        left = _elem("left", 10, 10, 40, 40)
        right = _elem("right", 60, 10, 90, 40)
        texts = {"left": "hello", "right": "world"}

        ordered = ReadingOrderResolver().resolve([left, right], page_width=100, texts=texts)

        # English text -> LTR -> left element first.
        assert [e.id for e in ordered] == ["left", "right"]

    def test_empty_elements_returns_empty(self) -> None:
        assert ReadingOrderResolver().resolve([], page_width=100) == []
