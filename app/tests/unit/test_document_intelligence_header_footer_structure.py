"""
Unit tests for cross-page header/footer detection with PAGE/NUMPAGES
field recognition (structure/header_footer.py)."""

import pytest

from apps.ocr.document_intelligence.ast import ASTNode, DocumentAST, PageAST
from apps.ocr.document_intelligence.layout import LayoutType
from apps.ocr.document_intelligence.structure.header_footer import (
    detect_header_footer_regions,
)


def _pages(nodes_per_page: dict[int, list[ASTNode]]) -> DocumentAST:
    return DocumentAST(
        pages=[
            PageAST(page_number=n, nodes=nodes)
            for n, nodes in sorted(nodes_per_page.items())
        ]
    )


@pytest.mark.document_intelligence
class TestPlainRepeatedText:
    def test_identical_footer_across_pages_promoted_as_plain_text(self) -> None:
        ast = _pages({
            n: [ASTNode(type=LayoutType.footer, text="Company Confidential")]
            for n in (1, 2, 3)
        })

        _headers, footers = detect_header_footer_regions(ast)

        assert len(footers) == 1
        assert footers[0].plain_text == "Company Confidential"
        assert not footers[0].has_page_field

    def test_text_seen_on_only_one_of_many_pages_not_promoted(self) -> None:
        ast = _pages(
            {1: [ASTNode(type=LayoutType.footer, text="Chapter 3 intro")]}
            | {n: [] for n in range(2, 6)}
        )

        _headers, footers = detect_header_footer_regions(ast)

        assert footers == []


@pytest.mark.document_intelligence
class TestPageNumberFieldRecognition:
    def test_footer_with_varying_page_number_becomes_page_field(self) -> None:
        ast = _pages({
            n: [ASTNode(type=LayoutType.footer, text=f"صفحه {n}")] for n in (1, 2, 3)
        })

        _headers, footers = detect_header_footer_regions(ast)

        assert len(footers) == 1
        region = footers[0]
        assert region.has_page_field
        kinds = [seg.kind for seg in region.segments]
        assert kinds == ["text", "page"]
        assert region.segments[0].text == "صفحه "

    def test_footer_with_page_and_total_becomes_page_and_numpages_fields(self) -> None:
        ast = _pages({
            n: [ASTNode(type=LayoutType.footer, text=f"صفحه {n} از 3")]
            for n in (1, 2, 3)
        })

        _headers, footers = detect_header_footer_regions(ast)

        region = footers[0]
        kinds = [seg.kind for seg in region.segments]
        assert kinds == ["text", "page", "text", "numpages"]

    def test_digit_that_never_changes_is_left_as_literal_text(self) -> None:
        """
        A constant number embedded in a repeating footer (e.g. a
        document ID) must not be misread as a page number just because it's
        a digit run — only a slot that tracks the page index every time
        qualifies."""
        ast = _pages({
            n: [ASTNode(type=LayoutType.footer, text="Doc #4471")] for n in (1, 2, 3)
        })

        _headers, footers = detect_header_footer_regions(ast)

        region = footers[0]
        assert not region.has_page_field
        assert region.plain_text == "Doc #4471"

    def test_standalone_page_number_nodes_promoted_to_page_field_footer(self) -> None:
        ast = _pages({
            n: [ASTNode(type=LayoutType.page_number, text=str(n))] for n in (1, 2, 3)
        })

        _headers, footers = detect_header_footer_regions(ast)

        assert len(footers) == 1
        assert footers[0].has_page_field

    def test_irregular_page_number_sequence_not_guessed(self) -> None:
        """
        If the digit doesn't cleanly equal the page index on every page
        (e.g. a scan artifact or an offset numbering scheme), don't guess —
        leave it unpromoted rather than emit a wrong PAGE field."""
        ast = _pages({
            1: [ASTNode(type=LayoutType.page_number, text="1")],
            2: [ASTNode(type=LayoutType.page_number, text="5")],
            3: [ASTNode(type=LayoutType.page_number, text="3")],
        })

        _headers, footers = detect_header_footer_regions(ast)

        assert footers == []

    def test_page_number_nodes_not_duplicated_when_footer_already_has_field(
        self,
    ) -> None:
        """
        When the footer text itself already embeds a verified page
        number, a separate page_number node sequence must not also add a
        second, redundant PAGE-field paragraph."""
        ast = _pages({
            n: [
                ASTNode(type=LayoutType.footer, text=f"صفحه {n}"),
                ASTNode(type=LayoutType.page_number, text=str(n)),
            ]
            for n in (1, 2, 3)
        })

        _headers, footers = detect_header_footer_regions(ast)

        assert len(footers) == 1


@pytest.mark.document_intelligence
class TestSinglePageDocument:
    def test_single_page_never_promotes_anything(self) -> None:
        ast = _pages({1: [ASTNode(type=LayoutType.footer, text="صفحه 1")]})

        headers, footers = detect_header_footer_regions(ast)

        assert headers == []
        assert footers == []
