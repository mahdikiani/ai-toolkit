"""Unit tests for the Document Intelligence Reading Order resolver."""

import pytest

from apps.ocr.document_intelligence.layout import LayoutElement, LayoutType
from apps.ocr.document_intelligence.reading_order import ReadingOrderResolver


def _elem(
    elem_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    elem_type: LayoutType = LayoutType.paragraph,
) -> LayoutElement:
    return LayoutElement(
        id=elem_id,
        page_id="p0001",
        page_number=1,
        type=elem_type,
        bbox=(x1, y1, x2, y2),
        padded_bbox=(x1, y1, x2, y2),
        confidence=0.9,
    )


@pytest.mark.document_intelligence
class TestDetectIsRtl:
    """Tests for auto language/direction detection."""

    def test_defaults_to_rtl_when_no_text(self) -> None:
        """No extracted text yet -> assume RTL (Persian/Arabic target documents)."""
        assert ReadingOrderResolver.detect_is_rtl([], None) is True
        assert ReadingOrderResolver.detect_is_rtl([], {}) is True

    def test_detects_rtl_for_persian_text(self) -> None:
        elem = _elem("e1", 0, 0, 10, 10)
        assert (
            ReadingOrderResolver.detect_is_rtl(
                [elem], {"e1": "سلام دنیا، این یک متن فارسی است"}
            )
            is True
        )

    def test_detects_ltr_for_english_text(self) -> None:
        elem = _elem("e1", 0, 0, 10, 10)
        assert (
            ReadingOrderResolver.detect_is_rtl(
                [elem], {"e1": "hello world this is english text"}
            )
            is False
        )


@pytest.mark.document_intelligence
class TestResolveFallbackOrdering:
    """Tests for the no-column fallback ordering path (< 3 candidate elements)."""

    def test_rtl_sorts_top_to_bottom_then_right_to_left(self) -> None:
        left = _elem("left", 10, 10, 40, 40)
        right = _elem("right", 60, 10, 90, 40)

        ordered = ReadingOrderResolver().resolve(
            [left, right], page_width=100, is_rtl=True
        )

        assert [e.id for e in ordered] == ["right", "left"]

    def test_ltr_sorts_top_to_bottom_then_left_to_right(self) -> None:
        left = _elem("left", 10, 10, 40, 40)
        right = _elem("right", 60, 10, 90, 40)

        ordered = ReadingOrderResolver().resolve(
            [left, right], page_width=100, is_rtl=False
        )

        assert [e.id for e in ordered] == ["left", "right"]

    def test_auto_detects_direction_from_texts_when_is_rtl_omitted(self) -> None:
        left = _elem("left", 10, 10, 40, 40)
        right = _elem("right", 60, 10, 90, 40)
        texts = {"left": "hello", "right": "world"}

        ordered = ReadingOrderResolver().resolve(
            [left, right], page_width=100, texts=texts
        )

        # English text -> LTR -> left element first.
        assert [e.id for e in ordered] == ["left", "right"]

    def test_empty_elements_returns_empty(self) -> None:
        assert ReadingOrderResolver().resolve([], page_width=100) == []


@pytest.mark.document_intelligence
class TestDetectFullWidth:
    """
    Regression: a heading/formula/etc. must be genuinely wide to be
    treated as full-width — type alone used to force it, which scrambled
    reading order for narrow headings sitting in a side-by-side box."""

    def test_narrow_heading_is_not_forced_full_width(self) -> None:
        narrow_heading = _elem(
            "h", 400, 0, 550, 30, elem_type=LayoutType.heading
        )  # 15% of 1000

        full_width = ReadingOrderResolver._detect_full_width(
            [narrow_heading], page_width=1000
        )

        assert full_width == []

    def test_reasonably_wide_heading_is_still_full_width(self) -> None:
        wide_heading = _elem(
            "h", 50, 0, 600, 30, elem_type=LayoutType.heading
        )  # 55% of 1000

        full_width = ReadingOrderResolver._detect_full_width(
            [wide_heading], page_width=1000
        )

        assert full_width == [wide_heading]

    def test_very_wide_paragraph_is_full_width_regardless_of_type(self) -> None:
        wide_paragraph = _elem("p", 10, 0, 900, 30)  # 89% of 1000, plain paragraph

        full_width = ReadingOrderResolver._detect_full_width(
            [wide_paragraph], page_width=1000
        )

        assert full_width == [wide_paragraph]

    def test_narrow_side_by_side_heading_now_sorts_with_its_column(self) -> None:
        """
        The exact real-world case: a narrow heading box on the right and
        a narrow paragraph box on the left, at the same row — in RTL, the
        right box should be read first."""
        right_heading = _elem("right", 550, 100, 950, 160, elem_type=LayoutType.heading)
        left_paragraph = _elem("left", 50, 100, 450, 160)
        # padding elements so a real column split can be detected (needs >=3
        # non-full-width candidates with a real x gap between groups)
        right_extra = _elem("right2", 560, 200, 940, 260)
        left_extra = _elem("left2", 60, 200, 440, 260)

        ordered = ReadingOrderResolver().resolve(
            [right_heading, left_paragraph, right_extra, left_extra],
            page_width=1000,
            is_rtl=True,
        )

        assert ordered[0].id == "right"


@pytest.mark.document_intelligence
class TestDetectColumns:
    """
    Regression: KMeans with a fixed n_clusters always returns that many
    clusters — even on an essentially single-column page — fabricating
    false columns and scrambling reading order. A real x-gap must exist
    before we treat a page as multi-column."""

    def test_evenly_spread_single_column_is_not_split(self) -> None:
        elements = [
            _elem("a", 100, 0, 140, 20),
            _elem("b", 150, 40, 190, 60),
            _elem("c", 200, 80, 240, 100),
            _elem("d", 250, 120, 290, 140),
            _elem("e", 300, 160, 340, 180),
        ]

        columns = ReadingOrderResolver._detect_columns(elements)

        assert columns == []

    def test_two_well_separated_clusters_are_split_into_columns(self) -> None:
        elements = [
            _elem("a1", 100, 0, 140, 20),
            _elem("a2", 110, 40, 150, 60),
            _elem("a3", 120, 80, 160, 100),
            _elem("b1", 400, 0, 440, 20),
            _elem("b2", 410, 40, 450, 60),
            _elem("b3", 420, 80, 460, 100),
        ]

        columns = ReadingOrderResolver._detect_columns(elements)

        assert len(columns) >= 2
        ids_by_column = [{e.id for e in col} for col in columns]
        assert any({"a1", "a2", "a3"} <= ids for ids in ids_by_column)

    def test_fewer_than_three_elements_never_splits(self) -> None:
        elements = [_elem("a", 0, 0, 10, 10), _elem("b", 500, 0, 510, 10)]
        assert ReadingOrderResolver._detect_columns(elements) == []


@pytest.mark.document_intelligence
class TestDetectColumnCount:
    def test_two_well_separated_clusters_report_two_columns(self) -> None:
        elements = [
            _elem("a1", 100, 0, 140, 20),
            _elem("a2", 110, 40, 150, 60),
            _elem("a3", 120, 80, 160, 100),
            _elem("b1", 400, 0, 440, 20),
            _elem("b2", 410, 40, 450, 60),
            _elem("b3", 420, 80, 460, 100),
        ]

        count = ReadingOrderResolver().detect_column_count(elements, page_width=500)

        assert count == 2

    def test_single_column_page_reports_one(self) -> None:
        elements = [
            _elem("a", 100, 0, 140, 20),
            _elem("b", 150, 40, 190, 60),
            _elem("c", 200, 80, 240, 100),
            _elem("d", 250, 120, 290, 140),
            _elem("e", 300, 160, 340, 180),
        ]

        count = ReadingOrderResolver().detect_column_count(elements, page_width=500)

        assert count == 1

    def test_no_elements_reports_one(self) -> None:
        count = ReadingOrderResolver().detect_column_count([], page_width=500)

        assert count == 1
