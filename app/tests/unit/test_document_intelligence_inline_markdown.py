"""Unit tests for inline Markdown segment parsing (inline_markdown.py)."""

import pytest

from apps.ocr.document_intelligence.inline_markdown import parse_inline_segments


@pytest.mark.document_intelligence
class TestParseInlineSegments:
    def test_plain_text_is_a_single_segment(self) -> None:
        segs = parse_inline_segments("just plain text")

        assert len(segs) == 1
        assert segs[0].text == "just plain text"
        assert not segs[0].bold and not segs[0].italic and not segs[0].code
        assert segs[0].url == ""

    def test_bold_markers_produce_a_bold_segment(self) -> None:
        segs = parse_inline_segments("a **bold** word")

        bold = [s for s in segs if s.bold]
        assert len(bold) == 1
        assert bold[0].text == "bold"

    def test_link_preserves_url(self) -> None:
        """
        The whole point of this behavior: a link must not be flattened
        to its visible text with the URL silently discarded."""
        segs = parse_inline_segments("see [our docs](https://example.com/x) now")

        links = [s for s in segs if s.url]
        assert len(links) == 1
        assert links[0].text == "our docs"
        assert links[0].url == "https://example.com/x"

    def test_link_segment_is_not_flagged_bold_italic_or_code(self) -> None:
        segs = parse_inline_segments("[text](https://example.com)")

        link = next(s for s in segs if s.url)
        assert not link.bold
        assert not link.italic
        assert not link.code

    def test_empty_text_returns_one_empty_segment(self) -> None:
        segs = parse_inline_segments("")

        assert len(segs) == 1
        assert segs[0].text == ""

    def test_mixed_styles_and_link_all_split_correctly(self) -> None:
        segs = parse_inline_segments("**bold** and `code` and [link](https://x.com/y) end")

        kinds = [
            "bold" if s.bold else "code" if s.code else "link" if s.url else "plain"
            for s in segs
            if s.text
        ]
        assert "bold" in kinds
        assert "code" in kinds
        assert "link" in kinds

    def test_single_dollar_math_span_is_flagged_math(self) -> None:
        """
        The whole point of this behavior: mid-sentence LaTeX must not be
        left as raw $...$ text sitting inside RTL Persian, where it
        reliably produces bidi word-reordering artifacts."""
        segs = parse_inline_segments("احتمال $p_i$ برابر است")

        math = [s for s in segs if s.math]
        assert len(math) == 1
        assert math[0].text == "p_i"

    def test_double_dollar_math_span_is_also_flagged_math(self) -> None:
        """
        Real VLM transcription of inline math is inconsistent about
        single vs double $ delimiters -- both must be caught."""
        segs = parse_inline_segments("مقدار $$w_i$$ را انتخاب کنید")

        math = [s for s in segs if s.math]
        assert len(math) == 1
        assert math[0].text == "w_i"

    def test_math_segment_is_not_flagged_bold_italic_code_or_url(self) -> None:
        segs = parse_inline_segments("$x^2$")

        math = next(s for s in segs if s.math)
        assert not math.bold
        assert not math.italic
        assert not math.code
        assert math.url == ""
