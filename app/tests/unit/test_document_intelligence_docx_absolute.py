"""
Unit tests for the absolute-layout Word renderer.

Every element gets its own floating text box positioned at its source-page
bbox, reconstructing the original page layout instead of a linear flow.
"""

from io import BytesIO

import pytest
from docx import Document as WordDocument
from docx.shared import Emu, Inches
from PIL import Image

from apps.ocr.document_intelligence.ast import ASTNode, DocumentAST, PageAST
from apps.ocr.document_intelligence.layout import LayoutType
from apps.ocr.document_intelligence.renderers.docx_absolute import render_docx_absolute


def _open(buf: BytesIO) -> WordDocument:
    buf.seek(0)
    return WordDocument(buf)


@pytest.mark.document_intelligence
class TestAbsolutePositioning:
    def test_node_positioned_at_bbox_converted_to_inches(self) -> None:
        # 200px at 200dpi -> 1.0in
        node = ASTNode(type=LayoutType.paragraph, text="hi", bbox=(200, 400, 600, 500))
        page = PageAST(
            page_number=1, nodes=[node], page_width=1000, page_height=1400, page_dpi=200
        )
        doc_ast = DocumentAST(pages=[page])

        xml = _open(render_docx_absolute(doc_ast)).element.xml

        assert "position:absolute" in xml
        assert "left:1.000in" in xml
        assert "top:2.000in" in xml
        assert "mso-position-horizontal-relative:page" in xml
        assert "mso-position-vertical-relative:page" in xml

    def test_missing_bbox_still_renders_with_minimum_size(self) -> None:
        node = ASTNode(type=LayoutType.paragraph, text="no bbox")
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        xml = _open(render_docx_absolute(doc_ast)).element.xml

        assert "position:absolute" in xml

    def test_page_break_inserted_between_pages(self) -> None:
        pages = [
            PageAST(
                page_number=1, nodes=[ASTNode(type=LayoutType.paragraph, text="p1")]
            ),
            PageAST(
                page_number=2, nodes=[ASTNode(type=LayoutType.paragraph, text="p2")]
            ),
        ]
        doc_ast = DocumentAST(pages=pages)

        xml = _open(render_docx_absolute(doc_ast)).element.xml

        assert '<w:br w:type="page"/>' in xml


@pytest.mark.document_intelligence
class TestContentInsideTextBoxes:
    def test_formula_renders_real_omml_inside_box(self) -> None:
        node = ASTNode(
            type=LayoutType.formula, latex=r"\frac{x^2}{y}", bbox=(0, 0, 200, 100)
        )
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        xml = _open(render_docx_absolute(doc_ast)).element.xml

        assert "oMath" in xml
        assert "w:txbxContent" in xml

    def test_table_renders_real_table_inside_box(self) -> None:
        node = ASTNode(
            type=LayoutType.table, rows=[["A", "B"], ["1", "2"]], bbox=(0, 0, 300, 200)
        )
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        xml = _open(render_docx_absolute(doc_ast)).element.xml

        assert "<w:tbl>" in xml
        assert ">A<" in xml
        assert ">2<" in xml

    def test_rtl_paragraph_marked_bidi(self) -> None:
        node = ASTNode(
            type=LayoutType.paragraph, text="سلام دنیا", bbox=(0, 0, 300, 100)
        )
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        xml = _open(render_docx_absolute(doc_ast)).element.xml

        assert "<w:bidi/>" in xml
        assert "سلام دنیا" in xml

    def test_list_renders_bulleted_items(self) -> None:
        node = ASTNode(
            type=LayoutType.list,
            bbox=(0, 0, 300, 200),
            children=[
                ASTNode(type=LayoutType.list, text="اول"),
                ASTNode(type=LayoutType.list, text="دوم"),
            ],
        )
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        xml = _open(render_docx_absolute(doc_ast)).element.xml

        assert "• اول" in xml
        assert "• دوم" in xml

    def test_header_footer_not_rendered_as_floating_boxes(self) -> None:
        """
        Headers/footers still go through the real Word section mechanism,
        not a floating box (they repeat per-page and aren't real content)."""
        node = ASTNode(
            type=LayoutType.header, text="Company Confidential", bbox=(0, 0, 300, 50)
        )
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        doc = _open(render_docx_absolute(doc_ast))

        assert doc.sections[0].header.paragraphs[0].text == "Company Confidential"
        assert "Company Confidential" not in doc.element.xml


@pytest.mark.document_intelligence
class TestImagePositioning:
    def test_image_embedded_with_absolute_position(self, tmp_path) -> None:
        img_path = tmp_path / "fig.png"
        Image.new("RGB", (50, 50), "white").save(img_path)

        node = ASTNode(
            type=LayoutType.figure, asset_path=str(img_path), bbox=(200, 400, 600, 800)
        )
        page = PageAST(
            page_number=1, nodes=[node], page_width=1000, page_height=1400, page_dpi=200
        )
        doc_ast = DocumentAST(pages=[page])

        doc = _open(render_docx_absolute(doc_ast))

        assert "v:imagedata" in doc.element.xml
        assert "left:1.000in" in doc.element.xml
        assert len(doc.part.related_parts) >= 1

    def test_missing_asset_is_skipped_without_crashing(self) -> None:
        node = ASTNode(
            type=LayoutType.figure,
            asset_path="/nonexistent/path.png",
            bbox=(0, 0, 100, 100),
        )
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        render_docx_absolute(doc_ast)  # must not raise


@pytest.mark.document_intelligence
class TestPageSizeAndFonts:
    def test_page_size_matches_source(self) -> None:
        page = PageAST(
            page_number=1, nodes=[], page_width=1000, page_height=1400, page_dpi=100
        )
        doc_ast = DocumentAST(pages=[page])

        doc = _open(render_docx_absolute(doc_ast))

        assert doc.sections[0].page_width == Emu(Inches(10.0))
        assert doc.sections[0].page_height == Emu(Inches(14.0))

    def test_output_round_trips_through_python_docx(self) -> None:
        """
        Basic sanity: whatever we build must be well-formed enough for
        python-docx (and by extension Word/LibreOffice) to open."""
        pages = [
            PageAST(
                page_number=1,
                page_width=1000,
                page_height=1400,
                page_dpi=200,
                nodes=[
                    ASTNode(type=LayoutType.title, text="Title", bbox=(0, 0, 500, 50)),
                    ASTNode(
                        type=LayoutType.paragraph, text="Body", bbox=(0, 60, 500, 120)
                    ),
                    ASTNode(
                        type=LayoutType.table,
                        rows=[["a", "b"]],
                        bbox=(0, 130, 300, 200),
                    ),
                    ASTNode(
                        type=LayoutType.formula, latex="x+y", bbox=(0, 210, 200, 260)
                    ),
                ],
            )
        ]
        doc_ast = DocumentAST(title="Doc", pages=pages)

        buf = render_docx_absolute(doc_ast)
        doc = _open(buf)  # must not raise
        assert doc.core_properties.title == "Doc"
