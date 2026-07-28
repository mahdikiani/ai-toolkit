"""Unit tests for apps.ocr.document_intelligence.elements."""

import pytest

from apps.ocr.document_intelligence.elements import _split_caption_description


@pytest.mark.document_intelligence
class TestSplitCaptionDescription:
    r"""
    A VLM's free-text response must never leak into the caption when it
    doesn't follow the requested "caption: ...\ndescription: ..." format
    -- caption renders as real visible text, description becomes image
    alt text, so an unparsed response must fall back to description-only.
    """

    def test_well_formed_response_splits_both_parts(self) -> None:
        text = "caption: Figure 1\ndescription: A scatter plot of two groups."
        caption, description = _split_caption_description(text)
        assert caption == "Figure 1"
        assert description == "A scatter plot of two groups."

    def test_capitalized_labels_still_split(self) -> None:
        """
        Regression: a previous version matched "description:" against
        a lowercased copy but sliced the original-cased text with it, so
        an actual "Description:" response silently failed to split."""
        text = "Caption: Figure 1\nDescription: A scatter plot of two groups."
        caption, description = _split_caption_description(text)
        assert caption == "Figure 1"
        assert description == "A scatter plot of two groups."

    def test_unparseable_response_becomes_description_only(self) -> None:
        """
        Regression: this exact free-text shape used to make caption
        AND description both equal the full text, which visibly leaked
        the whole alt-text-style blob into the document body via the
        (legitimately-rendered) caption field."""
        text = "A scatter plot showing two distinct groups of data points."
        caption, description = _split_caption_description(text)
        assert caption == ""
        assert description == text

    def test_caption_without_description_becomes_description_only(self) -> None:
        text = "caption: only a caption label, no description marker"
        caption, description = _split_caption_description(text)
        assert caption == ""
        assert description == text

    def test_description_before_caption_becomes_description_only(self) -> None:
        """
        Out-of-order labels aren't a format this parses -- treat as
        unparseable rather than guessing."""
        text = "description: the real description\ncaption: a label"
        caption, description = _split_caption_description(text)
        assert caption == ""
        assert description == text

    def test_overlong_caption_folds_into_description(self) -> None:
        """
        Regression: the VLM labeled a full sentence as "caption:"
        (technically valid format, but not a real caption) -- it must
        still not render as visible text just because it followed the
        two-field format the prompt asked for."""
        text = (
            "caption: A scatter plot showing two distinct classes of data "
            "points separated by a horizontal line.\n"
            "description: A 2D coordinate system with two clusters."
        )
        caption, description = _split_caption_description(text)
        assert caption == ""
        assert "scatter plot" in description
        assert "coordinate system" in description

    def test_short_caption_with_colon_still_counts_as_short(self) -> None:
        text = "caption: شکل ۱: داده‌های سوال\ndescription: a chart."
        caption, description = _split_caption_description(text)
        assert caption == "شکل ۱: داده‌های سوال"
        assert description == "a chart."
