"""Unit tests for the DocumentAST PDF renderer."""

from io import BytesIO

import fitz
import pytest

from apps.ocr.document_intelligence.ast import ASTNode, DocumentAST, PageAST
from apps.ocr.document_intelligence.layout import LayoutType
from apps.ocr.document_intelligence.markdown_parser import parse_markdown
from apps.ocr.document_intelligence.renderers.pdf import _render_table, render_pdf


def _open_pdf(markdown: str) -> fitz.Document:
    rendered = render_pdf(parse_markdown(markdown)).getvalue()
    return fitz.open(stream=rendered, filetype="pdf")


@pytest.mark.document_intelligence
class TestPdfRenderer:
    def test_structural_markdown_renders_valid_searchable_pdf(self) -> None:
        pdf = _open_pdf(
            """# Project Report

Introductory paragraph.

| Name | Value |
| --- | --- |
| Alpha | 42 |

- First item
- Second item
"""
        )

        assert pdf.page_count >= 1
        assert len(pdf.tobytes()) > 1_000
        text = "\n".join(page.get_text() for page in pdf)
        expected = [
            "Project Report",
            "Introductory paragraph.",
            "Name",
            "Value",
            "Alpha",
            "42",
            "First item",
            "Second item",
        ]
        positions = [text.index(value) for value in expected]
        assert positions == sorted(positions)

    def test_persian_text_round_trips_in_logical_order(self) -> None:
        phrase = "این یک متن فارسی برای آزمایش است"

        pdf = _open_pdf(f"# گزارش فارسی\n\n{phrase}")
        text = "\n".join(page.get_text() for page in pdf)

        assert "گزارش فارسی" in text
        assert phrase in text
        assert phrase[::-1] not in text

    def test_page_size_is_a4(self) -> None:
        pdf = _open_pdf("A4 document")
        page = pdf[0]

        assert page.rect.width == pytest.approx(595, abs=2)
        assert page.rect.height == pytest.approx(842, abs=2)

    def test_table_merges_emit_one_cell_per_merge_anchor(self) -> None:
        table = ASTNode(
            type=LayoutType.table,
            rows=[
                ["Merged heading", "", "Third"],
                ["Row span", "B", "C"],
                ["", "D", "E"],
            ],
            cell_merges=[(0, 0, 0, 1), (1, 0, 2, 0)],
        )
        fragment = _render_table(table)
        ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[table])])

        assert fragment.count("</th>") + fragment.count("</td>") == 7
        assert 'colspan="2"' in fragment
        assert 'rowspan="2"' in fragment

        pdf = fitz.open(stream=render_pdf(ast).getvalue(), filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
        assert text.count("Merged heading") == 1
        assert text.count("Row span") == 1

    def test_formula_fallback_code_and_optional_nodes_do_not_crash(self) -> None:
        nodes = [
            ASTNode(type=LayoutType.formula, html="x &lt; y"),
            ASTNode(type=LayoutType.code, text="if x < y:\n    return x"),
            ASTNode(type=LayoutType.figure, caption="Figure caption"),
            ASTNode(type=LayoutType.chart),
            ASTNode(type=LayoutType.header, text="Skipped header"),
            ASTNode(type=LayoutType.footer, text="Skipped footer"),
            ASTNode(type=LayoutType.page_number, text="1"),
        ]
        ast = DocumentAST(pages=[PageAST(page_number=1, nodes=nodes)])

        pdf = fitz.open(stream=render_pdf(ast).getvalue(), filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)

        assert "x &lt; y" in text
        assert "if x < y:" in text
        assert "Figure caption" in text
        assert "Skipped header" not in text

    def test_empty_and_malformed_tables_are_ignored_safely(self) -> None:
        empty = ASTNode(type=LayoutType.table)
        no_columns = ASTNode(type=LayoutType.table, rows=[[]])
        malformed_merge = ASTNode(
            type=LayoutType.table,
            rows=[["A"]],
            cell_merges=[(-1, 0, 8, 9), (0, 0, 0, 0)],
        )

        assert _render_table(empty) == ""
        assert _render_table(no_columns) == ""
        assert _render_table(malformed_merge).count("</th>") == 1

    def test_returns_seekable_bytes_io(self) -> None:
        result = render_pdf(parse_markdown("hello"))

        assert isinstance(result, BytesIO)
        assert result.tell() == 0
        assert result.read(5) == b"%PDF-"
