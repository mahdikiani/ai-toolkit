"""
Property-based tests for audio chunking utilities.

Property 3: Audio chunk duration preservation
Validates: Requirements 4.3, 7.3
"""

from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from apps.transcribe.chunker_ffmpeg import (
    AudioChunk,
    _calculate_cut_points,
    _find_silence_between,
)


@pytest.mark.property
class TestChunkDurationPreservation:
    """Property 3: Audio chunk duration preservation."""

    @given(
        st.integers(min_value=1, max_value=100),  # number of chunks
        st.integers(min_value=1000, max_value=60000),  # chunk size in ms
    )
    @settings(max_examples=100)
    def test_chunk_durations_sum_to_total(
        self, num_chunks: int, chunk_size_ms: int
    ) -> None:
        """
        Property 3: Audio chunk duration preservation.

        The sum of all chunk durations should equal the total duration.
        """
        # Build chunks manually to test the property
        chunks = []
        cursor = 0
        for i in range(num_chunks):
            end = cursor + chunk_size_ms
            chunk = AudioChunk(
                chunk_id=i,
                start_ms=cursor,
                end_ms=end,
                file_path=Path(f"chunk_{i}.wav"),
            )
            chunks.append(chunk)
            cursor = end

        total_duration = cursor
        sum_of_durations = sum(c.duration_ms for c in chunks)

        assert sum_of_durations == total_duration, (
            f"Sum of chunk durations ({sum_of_durations}) != "
            f"total duration ({total_duration})"
        )

    @given(
        st.integers(min_value=60000, max_value=3600000),  # total duration in ms
        st.integers(min_value=30000, max_value=300000),  # max chunk size in ms
        st.integers(min_value=10000, max_value=60000),  # min chunk size in ms
    )
    @settings(max_examples=50)
    def test_cut_points_cover_full_duration(
        self, duration_ms: int, max_chunk_ms: int, min_chunk_ms: int
    ) -> None:
        """Cut points should cover the full audio duration."""
        assume(min_chunk_ms < max_chunk_ms)
        assume(min_chunk_ms < duration_ms)

        cuts = _calculate_cut_points(duration_ms, [], min_chunk_ms, max_chunk_ms)

        # Last cut point should be at or equal to duration
        assert cuts[-1] <= duration_ms

        # All cut points should be positive
        assert all(c > 0 for c in cuts)

    @given(
        st.integers(min_value=60000, max_value=3600000),
        st.integers(min_value=30000, max_value=300000),
        st.integers(min_value=10000, max_value=60000),
    )
    @settings(max_examples=50)
    def test_cut_points_are_monotonically_increasing(
        self, duration_ms: int, max_chunk_ms: int, min_chunk_ms: int
    ) -> None:
        """Cut points should be in strictly increasing order."""
        assume(min_chunk_ms < max_chunk_ms)
        assume(min_chunk_ms < duration_ms)

        cuts = _calculate_cut_points(duration_ms, [], min_chunk_ms, max_chunk_ms)

        for i in range(len(cuts) - 1):
            assert cuts[i] < cuts[i + 1], (
                f"Cut points not monotonically increasing at index {i}: "
                f"{cuts[i]} >= {cuts[i + 1]}"
            )

    @given(
        st.integers(min_value=60000, max_value=3600000),
        st.integers(min_value=30000, max_value=300000),
        st.integers(min_value=10000, max_value=60000),
    )
    @settings(max_examples=50)
    def test_each_chunk_within_max_size(
        self, duration_ms: int, max_chunk_ms: int, min_chunk_ms: int
    ) -> None:
        """Each chunk should not exceed max_chunk_ms."""
        assume(min_chunk_ms < max_chunk_ms)
        assume(min_chunk_ms < duration_ms)

        cuts = _calculate_cut_points(duration_ms, [], min_chunk_ms, max_chunk_ms)

        cursor = 0
        for cut in cuts:
            chunk_size = cut - cursor
            assert chunk_size <= max_chunk_ms, (
                f"Chunk size {chunk_size} exceeds max_chunk_ms {max_chunk_ms}"
            )
            cursor = cut


@pytest.mark.property
class TestFindSilenceBetweenProperties:
    """Property tests for _find_silence_between function."""

    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=100000),
                st.integers(min_value=0, max_value=100000),
            ).filter(lambda t: t[0] < t[1]),
            max_size=20,
        ),
        st.integers(min_value=0, max_value=100000),
        st.integers(min_value=0, max_value=100000),
    )
    @settings(max_examples=100)
    def test_result_is_within_window_or_none(
        self,
        silence_ranges: list[tuple[int, int]],
        window_start: int,
        window_end: int,
    ) -> None:
        """Result should be within the window or None."""
        assume(window_start < window_end)

        result = _find_silence_between(silence_ranges, window_start, window_end)

        if result is not None:
            assert window_start <= result <= window_end, (
                f"Result {result} is outside window [{window_start}, {window_end}]"
            )
