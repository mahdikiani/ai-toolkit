"""Unit tests for the Document Intelligence Markdown renderer."""

import pytest

from apps.ocr.document_intelligence.ast import ASTNode, DocumentAST, PageAST
from apps.ocr.document_intelligence.layout import LayoutType
from apps.ocr.document_intelligence.renderers.markdown import (
    render_markdown,
    rewrite_asset_links,
)


def _doc(*nodes: ASTNode, assets: dict[str, str] | None = None) -> DocumentAST:
    return DocumentAST(
        pages=[PageAST(page_number=1, nodes=list(nodes))], assets=assets or {}
    )


@pytest.mark.document_intelligence
class TestRenderMarkdown:
    """Tests for rendering each ASTNode type to Markdown."""

    def test_title_renders_as_h1(self) -> None:
        md = render_markdown(_doc(ASTNode(type=LayoutType.title, text="Doc Title")))
        assert md.strip() == "# Doc Title"

    def test_heading_renders_as_h2(self) -> None:
        md = render_markdown(_doc(ASTNode(type=LayoutType.heading, text="Section")))
        assert md.strip() == "## Section"

    def test_paragraph_renders_plain(self) -> None:
        md = render_markdown(
            _doc(ASTNode(type=LayoutType.paragraph, text="hello world"))
        )
        assert md.strip() == "hello world"

    def test_table_renders_as_markdown_table(self) -> None:
        node = ASTNode(type=LayoutType.table, rows=[["A", "B"], ["1", "2"]])
        md = render_markdown(_doc(node))
        assert "| A | B |" in md
        assert "| --- | --- |" in md
        assert "| 1 | 2 |" in md

    def test_formula_renders_as_latex_block(self) -> None:
        node = ASTNode(type=LayoutType.formula, latex=r"\frac{x^2}{y}")
        md = render_markdown(_doc(node))
        assert md.strip() == "$$\n\\frac{x^2}{y}\n$$"

    def test_figure_renders_image_link_with_asset_mapping(self) -> None:
        node = ASTNode(type=LayoutType.figure, caption="a cat", asset_path="/tmp/x.png")
        md = render_markdown(_doc(node, assets={"/tmp/x.png": "assets/image_001.png"}))
        assert "![a cat](assets/image_001.png)" in md

    def test_code_renders_as_fenced_block(self) -> None:
        node = ASTNode(type=LayoutType.code, text="print(1)")
        md = render_markdown(_doc(node))
        assert md.strip() == "```\nprint(1)\n```"

    def test_reference_renders_as_blockquote(self) -> None:
        node = ASTNode(type=LayoutType.reference, text="Smith et al., 2020")
        md = render_markdown(_doc(node))
        assert md.strip() == "> Smith et al., 2020"

    def test_list_renders_children_as_bullets(self) -> None:
        node = ASTNode(
            type=LayoutType.list,
            children=[
                ASTNode(type=LayoutType.list, text="first"),
                ASTNode(type=LayoutType.list, text="second"),
            ],
        )
        md = render_markdown(_doc(node))
        assert "- first" in md
        assert "- second" in md

    def test_page_separator_between_multiple_pages(self) -> None:
        doc = DocumentAST(
            pages=[
                PageAST(
                    page_number=1, nodes=[ASTNode(type=LayoutType.paragraph, text="p1")]
                ),
                PageAST(
                    page_number=2, nodes=[ASTNode(type=LayoutType.paragraph, text="p2")]
                ),
            ]
        )
        md = render_markdown(doc)
        assert "---" in md


@pytest.mark.document_intelligence
class TestRewriteAssetLinks:
    """Tests for rewriting local asset paths to public URLs post-upload."""

    def test_replaces_known_local_path_with_url(self) -> None:
        md = "see ![cat](assets/image_001.png) above"
        result = rewrite_asset_links(
            md, {"assets/image_001.png": "https://cdn.example/img.png"}
        )
        assert result == "see ![cat](https://cdn.example/img.png) above"

    def test_leaves_unmapped_links_untouched(self) -> None:
        md = "see ![cat](assets/image_001.png) above"
        result = rewrite_asset_links(
            md, {"assets/other.png": "https://cdn.example/other.png"}
        )
        assert result == md
