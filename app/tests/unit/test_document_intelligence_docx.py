"""Unit tests for the Document Intelligence Word (DOCX) renderer."""

from io import BytesIO
from unittest.mock import patch

import pytest
from docx import Document as WordDocument
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Emu, Inches
from PIL import Image

from apps.ocr.document_intelligence.ast import ASTNode, DocumentAST, PageAST
from apps.ocr.document_intelligence.latex_omml import LatexConversionError
from apps.ocr.document_intelligence.layout import LayoutType
from apps.ocr.document_intelligence.renderers.docx import (
    _resolve_image_width_in,
    render_docx,
)


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


@pytest.mark.unit
class TestFontDetection:
    """The renderer must use the source PDF's real fonts when available,
    not always fall back to the fixed Calibri/B Nazanin defaults."""

    def test_detected_fonts_applied_to_normal_style(self) -> None:
        node = ASTNode(type=LayoutType.paragraph, text="hello")
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        with patch(
            "apps.ocr.pipeline.docx_renderer.detect_pdf_fonts",
            return_value={"cs": "Vazir", "latin": "Arial"},
        ):
            doc = _open(render_docx(doc_ast, pdf_data=b"%PDF-fake"))

        xml = doc.styles["Normal"].element.xml
        assert 'w:cs="Vazir"' in xml
        assert 'w:ascii="Arial"' in xml

    def test_no_pdf_data_uses_default_fonts(self) -> None:
        node = ASTNode(type=LayoutType.paragraph, text="hello")
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        doc = _open(render_docx(doc_ast))

        xml = doc.styles["Normal"].element.xml
        assert 'w:cs="B Nazanin"' in xml

    def test_font_detection_failure_falls_back_without_crashing(self) -> None:
        node = ASTNode(type=LayoutType.paragraph, text="hello")
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        with patch(
            "apps.ocr.pipeline.docx_renderer.detect_pdf_fonts",
            side_effect=RuntimeError("boom"),
        ):
            doc = _open(render_docx(doc_ast, pdf_data=b"%PDF-fake"))  # must not raise

        assert 'w:cs="B Nazanin"' in doc.styles["Normal"].element.xml


@pytest.mark.unit
class TestImageSizing:
    """Images must be sized relative to their footprint on the source page,
    not always dropped in at a fixed 5 inches."""

    def test_small_bbox_produces_small_width(self) -> None:
        width = _resolve_image_width_in(
            ASTNode(type=LayoutType.figure, bbox=(0, 0, 200, 200)),
            page_width_px=1000,
            content_width_in=6.0,
        )
        assert width == pytest.approx(1.5)  # 20% of content width, but floored at MIN

    def test_full_width_bbox_fills_content_width(self) -> None:
        width = _resolve_image_width_in(
            ASTNode(type=LayoutType.figure, bbox=(0, 0, 950, 200)),
            page_width_px=1000,
            content_width_in=6.0,
        )
        assert width == pytest.approx(5.7)

    def test_missing_bbox_falls_back_to_capped_default(self) -> None:
        width = _resolve_image_width_in(
            ASTNode(type=LayoutType.figure), page_width_px=1000, content_width_in=4.0
        )
        assert width == pytest.approx(4.0)  # capped to content width, not fixed 5"

    def test_rendered_image_width_matches_bbox_ratio(self, tmp_path) -> None:
        img_path = tmp_path / "fig.png"
        Image.new("RGB", (50, 50), "white").save(img_path)

        node = ASTNode(
            type=LayoutType.figure,
            asset_path=str(img_path),
            bbox=(0, 0, 500, 300),  # 50% of a 1000px-wide page
        )
        page = PageAST(page_number=1, nodes=[node], page_width=1000, page_height=1400, page_dpi=100)
        doc_ast = DocumentAST(pages=[page])

        doc = _open(render_docx(doc_ast))

        pic = doc.inline_shapes[0]
        # content width = 10in page (1000px/100dpi) minus 2*0.8in margin = 8.4in;
        # 50% of that = 4.2in
        assert pic.width == Emu(Inches(4.2))


@pytest.mark.unit
class TestParagraphAlignment:
    def test_centered_narrow_bbox_gets_center_alignment(self) -> None:
        node = ASTNode(type=LayoutType.paragraph, text="centered", bbox=(300, 0, 700, 30))
        page = PageAST(page_number=1, nodes=[node], page_width=1000, page_height=1400)
        doc_ast = DocumentAST(pages=[page])

        doc = _open(render_docx(doc_ast))

        matching = [p for p in doc.paragraphs if p.text == "centered"]
        assert matching
        assert matching[0].alignment == WD_ALIGN_PARAGRAPH.CENTER

    def test_off_center_bbox_stays_right_aligned(self) -> None:
        node = ASTNode(type=LayoutType.paragraph, text="right side", bbox=(700, 0, 950, 30))
        page = PageAST(page_number=1, nodes=[node], page_width=1000, page_height=1400)
        doc_ast = DocumentAST(pages=[page])

        doc = _open(render_docx(doc_ast))

        matching = [p for p in doc.paragraphs if p.text == "right side"]
        assert matching
        assert matching[0].alignment == WD_ALIGN_PARAGRAPH.RIGHT

    def test_missing_bbox_defaults_to_right_aligned(self) -> None:
        node = ASTNode(type=LayoutType.paragraph, text="no bbox")
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[node])])

        doc = _open(render_docx(doc_ast))

        matching = [p for p in doc.paragraphs if p.text == "no bbox"]
        assert matching
        assert matching[0].alignment == WD_ALIGN_PARAGRAPH.RIGHT


@pytest.mark.unit
class TestPageSize:
    def test_page_size_matches_source_page_dimensions(self) -> None:
        # 1000x1400 px at 100 dpi -> 10in x 14in
        page = PageAST(page_number=1, nodes=[], page_width=1000, page_height=1400, page_dpi=100)
        doc_ast = DocumentAST(pages=[page])

        doc = _open(render_docx(doc_ast))

        section = doc.sections[0]
        assert section.page_width == Emu(Inches(10.0))
        assert section.page_height == Emu(Inches(14.0))

    def test_missing_page_dims_fall_back_to_a4(self) -> None:
        doc_ast = DocumentAST(pages=[PageAST(page_number=1, nodes=[])])

        doc = _open(render_docx(doc_ast))

        section = doc.sections[0]
        # Section sizing round-trips through twips (1/20 pt) internally, so
        # a non-round inch value like 8.27" can be off by a few EMU.
        assert section.page_width.inches == pytest.approx(8.27, abs=0.01)
