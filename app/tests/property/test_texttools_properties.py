"""
Property-based tests for text normalization utilities.

Property 1: Text normalization idempotence
Validates: Requirements 4.2, 11.1
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from utils.texttools import normalize_text


@pytest.mark.property
class TestNormalizeTextProperties:
    """Property-based tests for normalize_text function."""

    @given(st.text())
    @settings(max_examples=100)
    def test_idempotence(self, text: str) -> None:
        """
        Property 1: Text normalization idempotence.

        normalize(normalize(x)) == normalize(x) for all text inputs.
        Applying normalization twice should produce the same result as once.
        """
        once = normalize_text(text)
        twice = normalize_text(once)
        assert once == twice, (
            f"normalize_text is not idempotent for input {text!r}:\n"
            f"  normalize(x)         = {once!r}\n"
            f"  normalize(normalize(x)) = {twice!r}"
        )

    @given(st.text(alphabet=st.characters(whitelist_categories=("L", "N", "P"))))
    @settings(max_examples=50)
    def test_idempotence_printable_chars(self, text: str) -> None:
        """Idempotence holds for printable characters."""
        once = normalize_text(text)
        twice = normalize_text(once)
        assert once == twice

    @given(st.text(alphabet="\r\n\t "))
    @settings(max_examples=50)
    def test_idempotence_whitespace_only(self, text: str) -> None:
        """Idempotence holds for whitespace-only strings."""
        once = normalize_text(text)
        twice = normalize_text(once)
        assert once == twice

    @given(st.text())
    @settings(max_examples=100)
    def test_result_has_no_crlf(self, text: str) -> None:
        """Normalized text should never contain CRLF sequences."""
        result = normalize_text(text)
        assert "\r\n" not in result, (
            f"normalize_text left CRLF in result for input {text!r}"
        )

    @given(st.text())
    @settings(max_examples=100)
    def test_result_has_no_leading_trailing_whitespace(self, text: str) -> None:
        """Normalized text should have no leading or trailing whitespace."""
        result = normalize_text(text)
        assert result == result.strip(), (
            f"normalize_text left leading/trailing whitespace for input {text!r}"
        )

    @given(st.text())
    @settings(max_examples=100)
    def test_length_is_non_increasing(self, text: str) -> None:
        """Normalized text should never be longer than the original."""
        result = normalize_text(text)
        assert len(result) <= len(text), (
            f"normalize_text increased length for input {text!r}: "
            f"{len(text)} -> {len(result)}"
        )
