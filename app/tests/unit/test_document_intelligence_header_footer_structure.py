"""
Unit tests for cross-page header/footer detection: position-aware
multi-slot promotion, PAGE/NUMPAGES field recognition, and Different First
Page detection (structure/header_footer.py)."""

import pytest

from apps.ocr.document_intelligence.ast import ASTNode, DocumentAST, PageAST
from apps.ocr.document_intelligence.layout import LayoutType
from apps.ocr.document_intelligence.structure.header_footer import (
    detect_header_footer_regions,
)


def _pages(nodes_per_page: dict[int, list[ASTNode]]) -> DocumentAST:
    return DocumentAST(
        pages=[
            PageAST(page_number=n, nodes=nodes, page_height=1000)
            for n, nodes in sorted(nodes_per_page.items())
        ]
    )


def _node(layout_type: LayoutType, text: str, y: float = 20.0) -> ASTNode:
    return ASTNode(type=layout_type, text=text, bbox=(0.0, y, 200.0, y + 30.0))


@pytest.mark.document_intelligence
class TestPlainRepeatedText:
    def test_identical_footer_across_pages_promoted_as_plain_text(self) -> None:
        ast = _pages(
            {n: [_node(LayoutType.footer, "Company Confidential")] for n in (1, 2, 3)}
        )

        _headers, footer_plan = detect_header_footer_regions(ast)

        assert footer_plan.promoted
        assert len(footer_plan.regions) == 1
        assert footer_plan.regions[0].plain_text == "Company Confidential"
        assert not footer_plan.regions[0].has_page_field

    def test_text_seen_on_only_one_of_many_pages_not_promoted(self) -> None:
        ast = _pages(
            {1: [_node(LayoutType.footer, "Chapter 3 intro")]}
            | {n: [] for n in range(2, 6)}
        )

        _headers, footer_plan = detect_header_footer_regions(ast)

        assert not footer_plan.promoted
        assert footer_plan.regions == ()


@pytest.mark.document_intelligence
class TestPageNumberFieldRecognition:
    def test_footer_with_varying_page_number_becomes_page_field(self) -> None:
        ast = _pages({n: [_node(LayoutType.footer, f"صفحه {n}")] for n in (1, 2, 3)})

        _headers, footer_plan = detect_header_footer_regions(ast)

        assert len(footer_plan.regions) == 1
        region = footer_plan.regions[0]
        assert region.has_page_field
        kinds = [seg.kind for seg in region.segments]
        assert kinds == ["text", "page"]
        assert region.segments[0].text == "صفحه "

    def test_footer_with_page_and_total_becomes_page_and_numpages_fields(self) -> None:
        ast = _pages(
            {n: [_node(LayoutType.footer, f"صفحه {n} از 3")] for n in (1, 2, 3)}
        )

        _headers, footer_plan = detect_header_footer_regions(ast)

        region = footer_plan.regions[0]
        kinds = [seg.kind for seg in region.segments]
        assert kinds == ["text", "page", "text", "numpages"]

    def test_digit_that_never_changes_is_left_as_literal_text(self) -> None:
        """
        A constant number embedded in a repeating footer (e.g. a
        document ID) must not be misread as a page number just because it's
        a digit run — only a slot that tracks the page index every time
        qualifies."""
        ast = _pages({n: [_node(LayoutType.footer, "Doc #4471")] for n in (1, 2, 3)})

        _headers, footer_plan = detect_header_footer_regions(ast)

        region = footer_plan.regions[0]
        assert not region.has_page_field
        assert region.plain_text == "Doc #4471"

    def test_standalone_page_number_nodes_promoted_to_page_field_footer(self) -> None:
        ast = _pages(
            {n: [_node(LayoutType.page_number, str(n))] for n in (1, 2, 3)}
        )

        _headers, footer_plan = detect_header_footer_regions(ast)

        assert len(footer_plan.regions) == 1
        assert footer_plan.regions[0].has_page_field

    def test_irregular_page_number_sequence_not_guessed(self) -> None:
        """
        If the digit doesn't cleanly equal the page index on every page
        (e.g. a scan artifact or an offset numbering scheme), don't guess —
        leave it unpromoted rather than emit a wrong PAGE field."""
        ast = _pages(
            {
                1: [_node(LayoutType.page_number, "1")],
                2: [_node(LayoutType.page_number, "5")],
                3: [_node(LayoutType.page_number, "3")],
            }
        )

        _headers, footer_plan = detect_header_footer_regions(ast)

        assert not footer_plan.promoted

    def test_page_number_nodes_not_duplicated_when_footer_already_has_field(
        self,
    ) -> None:
        """
        When the footer text itself already embeds a verified page
        number, a separate page_number node sequence must not also add a
        second, redundant PAGE-field paragraph."""
        ast = _pages(
            {
                n: [
                    _node(LayoutType.footer, f"صفحه {n}"),
                    _node(LayoutType.page_number, str(n)),
                ]
                for n in (1, 2, 3)
            }
        )

        _headers, footer_plan = detect_header_footer_regions(ast)

        assert len(footer_plan.regions) == 1


@pytest.mark.document_intelligence
class TestMultiSlotPromotion:
    """
    A page can have more than one independently-recurring header/footer
    element (e.g. a logo line above a separate banner line) -- each
    vertical slot must be checked and promoted on its own, not collapsed
    into a single "most common" winner."""

    def test_two_distinct_repeating_elements_at_different_positions_both_promoted(
        self,
    ) -> None:
        ast = _pages(
            {
                n: [
                    _node(LayoutType.header, "ACME Corp", y=10.0),
                    _node(LayoutType.header, "Confidential Draft", y=60.0),
                ]
                for n in (1, 2, 3)
            }
        )

        header_plan, _footer_plan = detect_header_footer_regions(ast)

        assert len(header_plan.regions) == 2
        texts = [r.plain_text for r in header_plan.regions]
        assert texts == ["ACME Corp", "Confidential Draft"]  # top-to-bottom order

    def test_only_one_slot_repeating_only_that_one_promoted(self) -> None:
        """
        The top line repeats every page; the lower line is page-specific
        (genuinely different, unrelated text each time -- not just a
        varying page number) and must not be promoted."""
        distinct_texts = {1: "Section Alpha", 2: "Section Beta", 3: "Section Gamma"}
        ast = _pages(
            {
                n: [
                    _node(LayoutType.header, "ACME Corp", y=10.0),
                    _node(LayoutType.header, distinct_texts[n], y=60.0),
                ]
                for n in (1, 2, 3)
            }
        )

        header_plan, _footer_plan = detect_header_footer_regions(ast)

        assert len(header_plan.regions) == 1
        assert header_plan.regions[0].plain_text == "ACME Corp"


@pytest.mark.document_intelligence
class TestDifferentFirstPage:
    def test_first_page_with_distinct_text_detected_as_variant(self) -> None:
        pages = {1: [_node(LayoutType.header, "Title Page")]}
        pages |= {n: [_node(LayoutType.header, "Chapter Header")] for n in (2, 3, 4)}
        ast = _pages(pages)

        header_plan, _footer_plan = detect_header_footer_regions(ast)

        assert header_plan.promoted
        assert header_plan.regions[0].plain_text == "Chapter Header"
        assert header_plan.different_first_page
        assert header_plan.first_page_regions[0].plain_text == "Title Page"

    def test_first_page_matching_the_regular_pattern_is_not_a_variant(self) -> None:
        ast = _pages({n: [_node(LayoutType.header, "Chapter Header")] for n in (1, 2, 3, 4)})

        header_plan, _footer_plan = detect_header_footer_regions(ast)

        assert header_plan.promoted
        assert not header_plan.different_first_page

    def test_no_established_regular_pattern_means_no_variant_either(self) -> None:
        """
        Without a verified regular pattern from pages 2..N, there's
        nothing for page 1 to differ from -- must not invent a "different
        first page" out of a single page's text alone."""
        ast = _pages({1: [_node(LayoutType.header, "Title Page")]} | {n: [] for n in (2, 3, 4)})

        header_plan, _footer_plan = detect_header_footer_regions(ast)

        assert not header_plan.promoted
        assert not header_plan.different_first_page

    def test_page1_absent_from_ast_means_no_variant(self) -> None:
        """
        Page 1 having no header/footer node at all (common for a title
        page) isn't evidence of a *distinct* first-page banner -- just
        absence -- so no variant is reported."""
        ast = _pages(
            {1: []} | {n: [_node(LayoutType.header, "Chapter Header")] for n in (2, 3, 4)}
        )

        header_plan, _footer_plan = detect_header_footer_regions(ast)

        assert header_plan.promoted
        assert not header_plan.different_first_page


@pytest.mark.document_intelligence
class TestSinglePageDocument:
    def test_single_page_never_promotes_anything(self) -> None:
        ast = _pages({1: [_node(LayoutType.footer, "صفحه 1")]})

        header_plan, footer_plan = detect_header_footer_regions(ast)

        assert not header_plan.promoted
        assert not footer_plan.promoted
