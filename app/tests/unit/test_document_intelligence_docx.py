"""Unit tests for the Document Intelligence Word (DOCX) renderer."""

from io import BytesIO
from unittest.mock import patch

import pytest
from docx import Document as WordDocument

from apps.ocr.document_intelligence.ast import ASTNode, DocumentAST, PageAST
from apps.ocr.document_intelligence.latex_omml import LatexConversionError
from apps.ocr.document_intelligence.layout import LayoutType
from apps.ocr.document_intelligence.renderers.docx import render_docx


def _open(buf: BytesIO) -> WordDocument:
    buf.seek(0)
    return WordDocument(buf)


@pytest.mark.unit
class TestFormulaRendersRealOmml:
    """Formulas must become real, editable OMML equation objects, not text."""

    def test_valid_formula_produces_omath_xml(self) -> None:
        doc_ast = DocumentAST(
            pages=[PageAST(page_number=1, nodes=[ASTNode(type=LayoutType.formula, latex=r"\frac{x^2}{y}")])]
        )

        buf = render_docx(doc_ast)
        doc = _open(buf)

        assert "oMath" in doc.element.xml

    def test_invalid_formula_falls_back_to_styled_text_without_crashing(self) -> None:
        doc_ast = DocumentAST(
            pages=[PageAST(page_number=1, nodes=[ASTNode(type=LayoutType.formula, latex="E=mc^2")])]
        )

        with patch(
            "apps.ocr.document_intelligence.renderers.docx.latex_to_omml",
            side_effect=LatexConversionError("boom"),
        ):
            buf = render_docx(doc_ast)  # must not raise
        doc = _open(buf)

        assert "oMath" not in doc.element.xml
        fallback_runs = [
            run
            for p in doc.paragraphs
            for run in p.runs
            if run.text == "E=mc^2"
        ]
        assert fallback_runs
        assert fallback_runs[0].font.name == "Cambria Math"


@pytest.mark.unit
class TestTableRendersRealWordTable:
    def test_rows_become_a_real_table_object(self) -> None:
        node = ASTNode(type=LayoutType.table, rows=[["A", "B"], ["1", "2"]])
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        doc = _open(render_docx(doc_ast))

        assert len(doc.tables) == 1
        table = doc.tables[0]
        assert table.cell(0, 0).text == "A"
        assert table.cell(1, 1).text == "2"


@pytest.mark.unit
class TestHeaderFooterUseRealWordSections:
    """Headers/footers must land in doc.sections[].header/footer, not the body."""

    def test_repeated_header_and_footer_promoted_to_section(self) -> None:
        pages = [
            PageAST(
                page_number=n,
                nodes=[
                    ASTNode(type=LayoutType.header, text="Company Confidential"),
                    ASTNode(type=LayoutType.paragraph, text=f"body {n}"),
                    ASTNode(type=LayoutType.footer, text="Page footer"),
                ],
            )
            for n in (1, 2)
        ]
        doc_ast = DocumentAST(pages=pages)

        doc = _open(render_docx(doc_ast))

        assert doc.sections[0].header.paragraphs[0].text == "Company Confidential"
        assert doc.sections[0].footer.paragraphs[0].text == "Page footer"
        body_text = "\n".join(p.text for p in doc.paragraphs)
        assert "Company Confidential" not in body_text
        assert "Page footer" not in body_text


@pytest.mark.unit
class TestRtlLayout:
    def test_paragraphs_are_marked_bidi(self) -> None:
        node = ASTNode(type=LayoutType.paragraph, text="سلام دنیا")
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        doc = _open(render_docx(doc_ast))
        body_paragraphs = [p for p in doc.paragraphs if p.text == "سلام دنیا"]

        assert body_paragraphs
        assert "w:bidi" in body_paragraphs[0]._p.xml


@pytest.mark.unit
class TestDocumentTitleMetadata:
    def test_title_set_on_core_properties(self) -> None:
        doc_ast = DocumentAST(title="My Report", pages=[])

        doc = _open(render_docx(doc_ast))

        assert doc.core_properties.title == "My Report"
