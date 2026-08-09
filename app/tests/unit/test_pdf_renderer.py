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

    def test_english_paragraph_is_left_aligned_not_forced_right(self) -> None:
        """
        A fully-English document must render left-aligned, not RTL.

        Regression: ``p { text-align: right }`` was hardcoded regardless
        of the paragraph's own ``dir`` -- per-character bidi ordering was
        already correct (see test_persian_text_round_trips_in_logical_
        order for that), but visually every paragraph, including pure
        English ones, was pushed flush to the page's right margin. Using
        the CSS logical value ``text-align: start`` (resolved from each
        element's own direction, not the page's blanket dir="rtl")
        fixes this: an English paragraph's short wrapped line must sit
        near the *left* margin, not the right one.
        """
        pdf = _open_pdf(
            "This is a long English paragraph that should wrap across at "
            "least two lines so we can check where the short final line "
            "lands on the page, left or right."
        )
        page = pdf[0]
        lines = [
            line
            for block in page.get_text("dict")["blocks"]
            if "lines" in block
            for line in block["lines"]
        ]
        assert len(lines) >= 2, "expected the paragraph to wrap"
        last_line = lines[-1]
        left_margin = 56.7  # 2cm page margin, in points
        assert last_line["bbox"][0] == pytest.approx(left_margin, abs=10), (
            f"last line should start near the left margin, got {last_line['bbox']}"
        )

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

    def test_wide_table_with_long_cells_stays_within_page_bounds(self) -> None:
        """
        A many-column table with long unbreakable values must not overflow.

        Regression, reported against a real 12-column bank statement
        export: ``table { width: 100% }`` alone doesn't cap width under
        auto table layout -- WeasyPrint (like browsers) sizes each column
        from its own content's natural width first, and the sum across
        many columns exceeded the page's printable width, pushing the
        table (and every cell after the first few columns) off the page.
        table-layout: fixed + overflow-wrap on cells fixes this by capping
        the table at its declared width and letting long values wrap.
        """
        header = [
            "تاریخ و زمان تراکنش", "نوع تراکنش", "مبلغ", "مانده حساب", "ارز",
            "شماره مرجع", "کد شعبه", "نام شعبه", "نام درخواست‌کننده",
            "نام خانوادگی", "کد هویتی",
        ]
        row = [
            "2026-02-18 11:46:37", "CREDIT (واریز)", "1.03", "501.03", "EUR",
            "17297581.0", "60.0", "مرکزی", "هژیر", "بی طوشی", "2872050256.0",
        ]
        table = ASTNode(type=LayoutType.table, rows=[header, row, row, row])
        ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[table])])

        pdf = fitz.open(stream=render_pdf(ast).getvalue(), filetype="pdf")
        page = pdf[0]
        # Neither the table's own borders/cell backgrounds (vector paths)
        # nor any text run may extend past the physical page -- table-
        # layout: fixed caps the table's box at its declared 100% width
        # regardless of what the columns' natural content widths wanted.
        for drawing in page.get_drawings():
            assert drawing["rect"].x1 <= page.rect.width + 1, (
                f"table border/fill extends past the page edge: {drawing['rect']}"
            )
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            x1 = block["bbox"][2]
            assert x1 <= page.rect.width + 1, (
                f"text block extends past the page edge: {block['bbox']}"
            )

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
