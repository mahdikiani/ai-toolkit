"""Unit tests for the Document Intelligence layout detector's dedup logic."""

import pytest

from apps.ocr.document_intelligence.layout import (
    LayoutElement,
    LayoutType,
    _iou,
    deduplicate_by_iou,
)


def _elem(elem_id: str, bbox: tuple[float, float, float, float]) -> LayoutElement:
    return LayoutElement(
        id=elem_id,
        page_id="p0001",
        page_number=1,
        type=LayoutType.paragraph,
        bbox=bbox,
        padded_bbox=bbox,
        confidence=0.9,
    )


@pytest.mark.document_intelligence
class TestDeduplicateByIou:
    def test_high_iou_boxes_are_deduped(self) -> None:
        a = _elem("a", (0, 0, 100, 100))
        b = _elem("b", (5, 5, 105, 105))

        kept = deduplicate_by_iou([a, b])

        assert len(kept) == 1

    def test_low_iou_but_high_containment_is_deduped(self) -> None:
        """
        v2/v3 ensemble: one model's box padded differently from the
        other's — low IOU, but one box is almost entirely inside the
        other. Regression: this used to survive dedup and double-OCR
        the same region."""
        outer = _elem("outer", (0, 0, 100, 100))
        inner = _elem("inner", (5, 5, 55, 55))  # fully inside `outer`, IOU=0.25

        assert _iou(outer.bbox, inner.bbox) < 0.40  # sanity: below plain IOU threshold

        kept = deduplicate_by_iou([outer, inner])

        assert len(kept) == 1
        assert kept[0].id == "outer"  # larger box kept

    def test_genuinely_separate_boxes_are_both_kept(self) -> None:
        left = _elem("left", (0, 0, 40, 40))
        right = _elem("right", (200, 0, 240, 40))

        kept = deduplicate_by_iou([left, right])

        assert len(kept) == 2

    def test_single_element_returned_as_is(self) -> None:
        a = _elem("a", (0, 0, 10, 10))
        assert deduplicate_by_iou([a]) == [a]
