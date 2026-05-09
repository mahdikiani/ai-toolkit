"""Unit tests for texttools utilities."""

import pytest

from utils.texttools import normalize_text


@pytest.mark.unit
class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_strips_leading_whitespace(self) -> None:
        """normalize_text should strip leading whitespace."""
        assert normalize_text("  hello") == "hello"

    def test_strips_trailing_whitespace(self) -> None:
        """normalize_text should strip trailing whitespace."""
        assert normalize_text("hello  ") == "hello"

    def test_strips_both_sides(self) -> None:
        """normalize_text should strip whitespace from both sides."""
        assert normalize_text("  hello  ") == "hello"

    def test_converts_crlf_to_lf(self) -> None:
        """normalize_text should convert CRLF line endings to LF."""
        assert normalize_text("line1\r\nline2") == "line1\nline2"

    def test_handles_multiple_crlf(self) -> None:
        """normalize_text should convert all CRLF occurrences."""
        result = normalize_text("a\r\nb\r\nc")
        assert result == "a\nb\nc"

    def test_preserves_lf_only(self) -> None:
        """normalize_text should preserve LF-only line endings."""
        assert normalize_text("line1\nline2") == "line1\nline2"

    def test_handles_empty_string(self) -> None:
        """normalize_text should handle empty strings."""
        assert normalize_text("") == ""

    def test_handles_whitespace_only(self) -> None:
        """normalize_text should return empty string for whitespace-only input."""
        assert normalize_text("   ") == ""

    def test_handles_tabs(self) -> None:
        """normalize_text should strip leading/trailing tabs."""
        assert normalize_text("\thello\t") == "hello"

    def test_preserves_internal_whitespace(self) -> None:
        """normalize_text should preserve internal whitespace."""
        result = normalize_text("  hello   world  ")
        assert result == "hello   world"

    def test_handles_unicode_text(self) -> None:
        """normalize_text should handle Unicode text correctly."""
        result = normalize_text("  سلام دنیا  ")
        assert result == "سلام دنیا"

    def test_handles_mixed_line_endings(self) -> None:
        """normalize_text should handle mixed CRLF and LF."""
        result = normalize_text("a\r\nb\nc\r\nd")
        assert result == "a\nb\nc\nd"

    def test_idempotent(self) -> None:
        """normalize_text should be idempotent (applying twice gives same result)."""
        text = "  hello\r\nworld  "
        once = normalize_text(text)
        twice = normalize_text(once)
        assert once == twice
