"""Unit tests for Paragraph Reconstruction (structure/paragraph_merge.py)."""

import pytest

from apps.ocr.document_intelligence.ast import ASTNode, DocumentAST, PageAST
from apps.ocr.document_intelligence.layout import LayoutType
from apps.ocr.document_intelligence.structure.paragraph_merge import merge_paragraphs

PAGE_WIDTH = 1000.0


def _para(text: str, bbox: tuple[float, float, float, float]) -> ASTNode:
    return ASTNode(type=LayoutType.paragraph, text=text, bbox=bbox, confidence=0.9)


def _doc(nodes: list[ASTNode], page_width: float = PAGE_WIDTH) -> DocumentAST:
    return DocumentAST(
        pages=[
            PageAST(page_number=1, nodes=nodes, page_width=page_width, page_height=1400)
        ]
    )


@pytest.mark.document_intelligence
class TestMergesGenuineContinuation:
    def test_incomplete_sentence_same_column_small_gap_merges(self) -> None:
        first = _para("این یک متن آزمایشی است که", (100, 0, 900, 40))
        second = _para("در دو خط نوشته شده است.", (100, 50, 900, 90))
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        nodes = merged.pages[0].nodes
        assert len(nodes) == 1
        assert nodes[0].text == "این یک متن آزمایشی است که در دو خط نوشته شده است."

    def test_hyphenated_line_break_joins_without_extra_space(self) -> None:
        first = _para("continu-", (100, 0, 900, 40))
        second = _para("ation", (100, 50, 900, 90))
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        assert merged.pages[0].nodes[0].text == "continu-ation"

    def test_merged_node_bbox_is_the_union(self) -> None:
        first = _para("part one", (100, 10, 900, 40))
        second = _para("part two", (100, 50, 900, 90))
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        assert merged.pages[0].nodes[0].bbox == (100, 10, 900, 90)

    def test_three_blocks_all_merge_into_one(self) -> None:
        blocks = [
            _para("one", (100, 0, 900, 40)),
            _para("two", (100, 50, 900, 90)),
            _para("three.", (100, 100, 900, 140)),
        ]
        doc_ast = _doc(blocks)

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 1
        assert merged.pages[0].nodes[0].text == "one two three."


@pytest.mark.document_intelligence
class TestDoesNotMerge:
    def test_sentence_ending_in_period_stays_separate(self) -> None:
        first = _para("This is a complete sentence.", (100, 0, 900, 40))
        second = _para("This starts a new paragraph.", (100, 50, 900, 90))
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 2

    def test_persian_question_mark_stays_separate(self) -> None:
        first = _para("این سوال است؟", (100, 0, 900, 40))
        second = _para("این جواب است.", (100, 50, 900, 90))
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 2

    def test_next_block_starting_with_bullet_stays_separate(self) -> None:
        first = _para("Items below", (100, 0, 900, 40))
        second = _para("- first item", (100, 50, 900, 90))
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 2

    def test_different_column_left_edge_stays_separate(self) -> None:
        """
        Same page, but the second block starts at a very different left
        edge -- e.g. two side-by-side columns, not a continuation."""
        first = _para("Left column text that", (50, 0, 450, 40))
        second = _para("Right column text here", (550, 0, 950, 40))
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 2

    def test_large_vertical_gap_stays_separate(self) -> None:
        """
        A gap much larger than the block's own height reads as a real
        paragraph break (space_after), not mid-paragraph line spacing."""
        first = _para("First paragraph text here", (100, 0, 900, 40))
        second = _para("A separate paragraph far below", (100, 400, 900, 440))
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 2

    def test_narrow_block_after_full_width_block_stays_separate(self) -> None:
        """
        Width mismatch -- e.g. a full-width paragraph followed by a
        narrow caption that merely happens to share a left edge."""
        first = _para("Full width paragraph text", (100, 0, 900, 40))
        second = _para("caption", (100, 50, 250, 90))
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 2

    def test_non_paragraph_neighbor_never_merged(self) -> None:
        heading = ASTNode(
            type=LayoutType.heading, text="Section Title", bbox=(100, 0, 900, 40)
        )
        para = _para("Body text follows", (100, 50, 900, 90))
        doc_ast = _doc([heading, para])

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 2
        assert merged.pages[0].nodes[0].type == LayoutType.heading

    def test_missing_bbox_never_merged(self) -> None:
        first = ASTNode(type=LayoutType.paragraph, text="no bbox here")
        second = ASTNode(type=LayoutType.paragraph, text="also no bbox")
        doc_ast = _doc([first, second])

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 2

    def test_cross_page_paragraphs_never_merged(self) -> None:
        """Scoped to same-page merging only -- see module docstring."""
        page1 = PageAST(
            page_number=1,
            nodes=[_para("continues on next page", (100, 900, 900, 940))],
            page_width=PAGE_WIDTH,
            page_height=1000,
        )
        page2 = PageAST(
            page_number=2,
            nodes=[_para("the rest of it.", (100, 0, 900, 40))],
            page_width=PAGE_WIDTH,
            page_height=1000,
        )
        doc_ast = DocumentAST(pages=[page1, page2])

        merged = merge_paragraphs(doc_ast)

        assert len(merged.pages[0].nodes) == 1
        assert len(merged.pages[1].nodes) == 1
