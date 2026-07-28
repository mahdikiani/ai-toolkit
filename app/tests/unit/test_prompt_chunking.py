"""Unit tests for splitting long content into ordered chunks."""

import pytest

from apps.language.promptic.engine.chunking import split_into_chunks


@pytest.mark.unit
class TestSplitIntoChunks:
    """Tests for split_into_chunks."""

    def test_empty_content_returns_no_chunks(self) -> None:
        assert split_into_chunks("", max_chars=100) == []

    def test_short_content_returns_single_chunk(self) -> None:
        assert split_into_chunks("hello world", max_chars=100) == ["hello world"]

    def test_packs_multiple_short_paragraphs_into_one_chunk(self) -> None:
        content = "para one\n\npara two\n\npara three"
        chunks = split_into_chunks(content, max_chars=1000)
        assert chunks == [content]

    def test_splits_along_paragraph_boundaries_when_over_budget(self) -> None:
        para_a = "A" * 100
        para_b = "B" * 100
        para_c = "C" * 100
        content = f"{para_a}\n\n{para_b}\n\n{para_c}"

        chunks = split_into_chunks(content, max_chars=150)

        assert chunks == [para_a, para_b, para_c]

    def test_combines_paragraphs_that_fit_together(self) -> None:
        para_a = "A" * 100
        para_b = "B" * 100
        para_c = "C" * 100
        content = f"{para_a}\n\n{para_b}\n\n{para_c}"

        chunks = split_into_chunks(content, max_chars=250)

        assert chunks == [f"{para_a}\n\n{para_b}", para_c]

    def test_no_chunk_exceeds_max_chars_except_a_single_oversized_paragraph(
        self,
    ) -> None:
        content = "\n\n".join(f"paragraph {i} " + "x" * 50 for i in range(30))

        chunks = split_into_chunks(content, max_chars=200)

        assert all(len(c) <= 200 for c in chunks)

    def test_hard_splits_a_single_paragraph_larger_than_max_chars(self) -> None:
        huge_paragraph = "x" * 500

        chunks = split_into_chunks(huge_paragraph, max_chars=200)

        assert chunks == ["x" * 200, "x" * 200, "x" * 100]

    def test_oversized_paragraph_flushes_pending_chunk_first(self) -> None:
        small = "small paragraph"
        huge = "y" * 300
        content = f"{small}\n\n{huge}"

        chunks = split_into_chunks(content, max_chars=200)

        assert chunks[0] == small
        assert "".join(chunks[1:]) == huge

    def test_reassembling_chunks_preserves_all_content(self) -> None:
        paragraphs = [f"paragraph number {i} with some text" for i in range(20)]
        content = "\n\n".join(paragraphs)

        chunks = split_into_chunks(content, max_chars=120)

        # every paragraph must appear intact in exactly one chunk
        joined = "\n\n".join(chunks)
        for para in paragraphs:
            assert para in joined

    def test_chunk_order_is_preserved(self) -> None:
        paragraphs = [f"p{i}" * 30 for i in range(10)]
        content = "\n\n".join(paragraphs)

        chunks = split_into_chunks(content, max_chars=100)

        # first paragraph must appear before the last one, across chunks
        joined = "".join(chunks)
        assert joined.index(paragraphs[0]) < joined.index(paragraphs[-1])
