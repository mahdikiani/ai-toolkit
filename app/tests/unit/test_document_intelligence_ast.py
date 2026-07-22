"""Unit tests for the Document Intelligence AST builder."""

import pytest

from apps.ocr.document_intelligence.ast import build_ast, build_document_ast
from apps.ocr.document_intelligence.elements import ProcessedElement
from apps.ocr.document_intelligence.layout import LayoutElement, LayoutType


def _layout_elem(elem_id: str, elem_type: LayoutType, x1: float = 0.0) -> LayoutElement:
    return LayoutElement(
        id=elem_id,
        page_id="p0001",
        page_number=1,
        type=elem_type,
        bbox=(x1, 0.0, x1 + 10, 10.0),
        padded_bbox=(x1, 0.0, x1 + 10, 10.0),
        confidence=0.9,
    )


@pytest.mark.unit
class TestBuildAst:
    """Tests for build_ast — processed elements + reading order -> PageAST."""

    def test_orders_nodes_by_reading_order(self) -> None:
        """Nodes must follow the order of the `ordered` list, not `processed`."""
        e1 = ProcessedElement(id="e1", type=LayoutType.paragraph, text="second")
        e2 = ProcessedElement(id="e2", type=LayoutType.paragraph, text="first")
        ordered = [_layout_elem("e2", LayoutType.paragraph), _layout_elem("e1", LayoutType.paragraph)]

        page_ast = build_ast([e1, e2], ordered, page_number=3)

        assert page_ast.page_number == 3
        assert [n.text for n in page_ast.nodes] == ["first", "second"]

    def test_title_gets_heading_level_one(self) -> None:
        """Title elements are promoted to heading level 1."""
        e1 = ProcessedElement(id="e1", type=LayoutType.title, text="Doc Title")
        ordered = [_layout_elem("e1", LayoutType.title)]

        page_ast = build_ast([e1], ordered, page_number=1)

        assert page_ast.nodes[0].level == 1

    def test_table_html_converted_to_rows(self) -> None:
        """Table nodes get `.rows` parsed out of their HTML."""
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        e1 = ProcessedElement(id="e1", type=LayoutType.table, text=html, html=html)
        ordered = [_layout_elem("e1", LayoutType.table)]

        page_ast = build_ast([e1], ordered, page_number=1)

        assert page_ast.nodes[0].rows == [["A", "B"], ["1", "2"]]

    def test_list_text_split_into_children(self) -> None:
        """List nodes split their raw text into one child ASTNode per line."""
        e1 = ProcessedElement(id="e1", type=LayoutType.list, text="one\ntwo\nthree")
        ordered = [_layout_elem("e1", LayoutType.list)]

        page_ast = build_ast([e1], ordered, page_number=1)

        assert [c.text for c in page_ast.nodes[0].children] == ["one", "two", "three"]

    def test_unmapped_element_falls_back_to_end(self) -> None:
        """A processed element missing from `ordered` sorts after ordered ones."""
        e1 = ProcessedElement(id="known", type=LayoutType.paragraph, text="known")
        e2 = ProcessedElement(id="orphan", type=LayoutType.paragraph, text="orphan")
        ordered = [_layout_elem("known", LayoutType.paragraph)]

        page_ast = build_ast([e2, e1], ordered, page_number=1)

        assert [n.text for n in page_ast.nodes] == ["known", "orphan"]


@pytest.mark.unit
class TestBuildDocumentAst:
    """Tests for build_document_ast — combining pages into a DocumentAST."""

    def test_combines_all_pages_in_order(self) -> None:
        e1 = ProcessedElement(id="e1", type=LayoutType.paragraph, text="p1")
        e2 = ProcessedElement(id="e2", type=LayoutType.paragraph, text="p2")
        page1 = build_ast([e1], [_layout_elem("e1", LayoutType.paragraph)], page_number=1)
        page2 = build_ast([e2], [_layout_elem("e2", LayoutType.paragraph)], page_number=2)

        doc_ast = build_document_ast([page1, page2])

        assert [p.page_number for p in doc_ast.pages] == [1, 2]

    def test_title_pulled_from_first_title_node(self) -> None:
        title_elem = ProcessedElement(id="t1", type=LayoutType.title, text="  My Document  ")
        page1 = build_ast([title_elem], [_layout_elem("t1", LayoutType.title)], page_number=1)

        doc_ast = build_document_ast([page1])

        assert doc_ast.title == "My Document"

    def test_no_title_node_leaves_title_empty(self) -> None:
        e1 = ProcessedElement(id="e1", type=LayoutType.paragraph, text="just text")
        page1 = build_ast([e1], [_layout_elem("e1", LayoutType.paragraph)], page_number=1)

        doc_ast = build_document_ast([page1])

        assert doc_ast.title == ""

    def test_asset_map_is_carried_through(self) -> None:
        doc_ast = build_document_ast([], asset_map={"/tmp/x.png": "assets/x.png"})

        assert doc_ast.assets == {"/tmp/x.png": "assets/x.png"}
