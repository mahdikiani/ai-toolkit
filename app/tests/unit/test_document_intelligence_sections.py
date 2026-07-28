"""Unit tests for Section Detection (structure/sections.py)."""

import pytest

from apps.ocr.document_intelligence.ast import DocumentAST, PageAST
from apps.ocr.document_intelligence.structure.sections import detect_sections


def _page(
    n: int, width: float, height: float, dpi: float = 100.0, column_count: int = 1
) -> PageAST:
    return PageAST(
        page_number=n,
        nodes=[],
        page_width=width,
        page_height=height,
        page_dpi=dpi,
        column_count=column_count,
    )


@pytest.mark.document_intelligence
class TestDetectSections:
    def test_uniform_pages_form_a_single_section(self) -> None:
        ast = DocumentAST(pages=[_page(n, 1000, 1400) for n in (1, 2, 3)])

        spans = detect_sections(ast)

        assert len(spans) == 1
        assert spans[0].start_page_number == 1
        assert spans[0].end_page_number == 3

    def test_landscape_page_in_the_middle_starts_a_new_section(self) -> None:
        pages = [
            _page(1, 1000, 1400),
            _page(2, 1000, 1400),
            _page(3, 1400, 1000),  # landscape
            _page(4, 1000, 1400),
        ]
        ast = DocumentAST(pages=pages)

        spans = detect_sections(ast)

        assert len(spans) == 3
        assert (spans[0].start_page_number, spans[0].end_page_number) == (1, 2)
        assert (spans[1].start_page_number, spans[1].end_page_number) == (3, 3)
        assert (spans[2].start_page_number, spans[2].end_page_number) == (4, 4)
        assert spans[1].page_width_in > spans[1].page_height_in  # landscape

    def test_small_dpi_rounding_jitter_does_not_split_a_section(self) -> None:
        pages = [_page(1, 1000, 1400), _page(2, 1002, 1399), _page(3, 999, 1401)]
        ast = DocumentAST(pages=pages)

        spans = detect_sections(ast)

        assert len(spans) == 1

    def test_empty_document_returns_one_placeholder_span(self) -> None:
        ast = DocumentAST(pages=[])

        spans = detect_sections(ast)

        assert len(spans) == 1

    def test_unknown_size_pages_do_not_force_a_new_section(self) -> None:
        pages = [_page(1, 1000, 1400), _page(2, 0, 0), _page(3, 1000, 1400)]
        ast = DocumentAST(pages=pages)

        spans = detect_sections(ast)

        assert len(spans) == 1

    def test_column_count_change_starts_a_new_section(self) -> None:
        pages = [
            _page(1, 1000, 1400, column_count=1),
            _page(2, 1000, 1400, column_count=2),
            _page(3, 1000, 1400, column_count=2),
            _page(4, 1000, 1400, column_count=1),
        ]
        ast = DocumentAST(pages=pages)

        spans = detect_sections(ast)

        assert len(spans) == 3
        assert (spans[0].start_page_number, spans[0].end_page_number) == (1, 1)
        assert spans[0].column_count == 1
        assert (spans[1].start_page_number, spans[1].end_page_number) == (2, 3)
        assert spans[1].column_count == 2
        assert (spans[2].start_page_number, spans[2].end_page_number) == (4, 4)
        assert spans[2].column_count == 1
