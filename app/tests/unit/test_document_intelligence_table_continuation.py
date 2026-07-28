"""Unit tests for Table Continuation (structure/table_continuation.py)."""

import pytest

from apps.ocr.document_intelligence.ast import ASTNode, DocumentAST, PageAST
from apps.ocr.document_intelligence.layout import LayoutType
from apps.ocr.document_intelligence.structure.table_continuation import (
    merge_table_continuations,
)

PAGE_HEIGHT = 1000.0


def _table(rows: list[list[str]], y1: float, y2: float) -> ASTNode:
    return ASTNode(type=LayoutType.table, rows=rows, bbox=(0.0, y1, 500.0, y2))


def _page(n: int, nodes: list[ASTNode]) -> PageAST:
    return PageAST(page_number=n, nodes=nodes, page_height=PAGE_HEIGHT)


@pytest.mark.document_intelligence
class TestMergesGenuineContinuation:
    def test_matching_columns_near_page_edges_merges(self) -> None:
        first = _table([["A", "B"], ["1", "2"]], y1=100, y2=950)
        continuation = _table([["3", "4"]], y1=20, y2=200)
        ast = DocumentAST(pages=[_page(1, [first]), _page(2, [continuation])])

        merged = merge_table_continuations(ast)

        assert len(merged.pages[0].nodes) == 1
        assert len(merged.pages[1].nodes) == 0
        table_node = merged.pages[0].nodes[0]
        assert table_node.rows == [["A", "B"], ["1", "2"], ["3", "4"]]
        assert table_node.repeat_header_row is True

    def test_duplicated_header_row_on_continuation_is_dropped(self) -> None:
        first = _table([["A", "B"], ["1", "2"]], y1=100, y2=950)
        continuation = _table([["A", "B"], ["3", "4"]], y1=20, y2=200)
        ast = DocumentAST(pages=[_page(1, [first]), _page(2, [continuation])])

        merged = merge_table_continuations(ast)

        table_node = merged.pages[0].nodes[0]
        assert table_node.rows == [["A", "B"], ["1", "2"], ["3", "4"]]

    def test_cell_merges_from_continuation_offset_correctly(self) -> None:
        first = ASTNode(
            type=LayoutType.table,
            rows=[["A", "B"], ["1", "2"]],
            cell_merges=[],
            bbox=(0.0, 100.0, 500.0, 950.0),
        )
        continuation = ASTNode(
            type=LayoutType.table,
            rows=[["Span", ""]],
            cell_merges=[(0, 0, 0, 1)],
            bbox=(0.0, 20.0, 500.0, 200.0),
        )
        ast = DocumentAST(pages=[_page(1, [first]), _page(2, [continuation])])

        merged = merge_table_continuations(ast)

        table_node = merged.pages[0].nodes[0]
        # continuation's row 0 lands at merged row index 2 (first has 2 rows).
        assert table_node.cell_merges == [(2, 0, 2, 1)]

    def test_other_nodes_on_either_page_are_preserved(self) -> None:
        heading = ASTNode(type=LayoutType.heading, text="Report")
        first = _table([["A", "B"], ["1", "2"]], y1=100, y2=950)
        continuation = _table([["3", "4"]], y1=20, y2=200)
        footer_para = ASTNode(type=LayoutType.paragraph, text="notes")
        ast = DocumentAST(
            pages=[_page(1, [heading, first]), _page(2, [continuation, footer_para])]
        )

        merged = merge_table_continuations(ast)

        assert [n.type for n in merged.pages[0].nodes] == [LayoutType.heading, LayoutType.table]
        assert [n.type for n in merged.pages[1].nodes] == [LayoutType.paragraph]


@pytest.mark.document_intelligence
class TestDoesNotMerge:
    def test_different_column_counts_not_merged(self) -> None:
        first = _table([["A", "B"], ["1", "2"]], y1=100, y2=950)
        continuation = _table([["3", "4", "5"]], y1=20, y2=200)
        ast = DocumentAST(pages=[_page(1, [first]), _page(2, [continuation])])

        merged = merge_table_continuations(ast)

        assert len(merged.pages[0].nodes) == 1
        assert len(merged.pages[1].nodes) == 1

    def test_first_table_not_near_bottom_of_page_not_merged(self) -> None:
        """
        A table that ends well before the bottom of the page isn't
        cut off -- it's just a normal, complete table."""
        first = _table([["A", "B"], ["1", "2"]], y1=100, y2=400)
        continuation = _table([["3", "4"]], y1=20, y2=200)
        ast = DocumentAST(pages=[_page(1, [first]), _page(2, [continuation])])

        merged = merge_table_continuations(ast)

        assert len(merged.pages[0].nodes) == 1
        assert len(merged.pages[1].nodes) == 1

    def test_continuation_not_near_top_of_next_page_not_merged(self) -> None:
        first = _table([["A", "B"], ["1", "2"]], y1=100, y2=950)
        continuation = _table([["3", "4"]], y1=600, y2=800)
        ast = DocumentAST(pages=[_page(1, [first]), _page(2, [continuation])])

        merged = merge_table_continuations(ast)

        assert len(merged.pages[0].nodes) == 1
        assert len(merged.pages[1].nodes) == 1

    def test_non_table_neighbor_never_merged(self) -> None:
        first = _table([["A", "B"], ["1", "2"]], y1=100, y2=950)
        paragraph = ASTNode(type=LayoutType.paragraph, text="not a table", bbox=(0, 20, 500, 200))
        ast = DocumentAST(pages=[_page(1, [first]), _page(2, [paragraph])])

        merged = merge_table_continuations(ast)

        assert len(merged.pages[0].nodes) == 1
        assert len(merged.pages[1].nodes) == 1

    def test_empty_table_rows_never_merged(self) -> None:
        first = ASTNode(type=LayoutType.table, rows=[], bbox=(0, 100, 500, 950))
        continuation = _table([["3", "4"]], y1=20, y2=200)
        ast = DocumentAST(pages=[_page(1, [first]), _page(2, [continuation])])

        merged = merge_table_continuations(ast)

        assert len(merged.pages[0].nodes) == 1
        assert len(merged.pages[1].nodes) == 1

    def test_non_adjacent_pages_never_considered(self) -> None:
        """
        Only a genuinely adjacent page pair is ever checked -- this is
        implicit in the loop bounds, exercised here via a 3-page document
        where pages 1 and 3 would otherwise "match" if compared."""
        first = _table([["A", "B"], ["1", "2"]], y1=100, y2=950)
        middle = ASTNode(type=LayoutType.paragraph, text="unrelated page")
        continuation = _table([["3", "4"]], y1=20, y2=200)
        ast = DocumentAST(pages=[_page(1, [first]), _page(2, [middle]), _page(3, [continuation])])

        merged = merge_table_continuations(ast)

        assert len(merged.pages[0].nodes) == 1
        assert len(merged.pages[2].nodes) == 1
