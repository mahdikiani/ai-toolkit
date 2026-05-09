"""Unit tests for file processing utilities."""

from pathlib import Path

import pytest

from apps.transcribe.chunker_ffmpeg import (
    AudioChunk,
    ChunkPlan,
    _calculate_cut_points,
    _find_silence_between,
    _guess_extension,
)


@pytest.mark.unit
class TestAudioChunk:
    """Tests for AudioChunk model."""

    def test_duration_ms_property(self) -> None:
        """AudioChunk.duration_ms should return end_ms - start_ms."""
        chunk = AudioChunk(
            chunk_id=0,
            start_ms=1000,
            end_ms=6000,
            file_path=Path("chunk.wav"),
        )
        assert chunk.duration_ms == 5000

    def test_zero_duration(self) -> None:
        """AudioChunk.duration_ms should return 0 for same start and end."""
        chunk = AudioChunk(
            chunk_id=0,
            start_ms=5000,
            end_ms=5000,
            file_path=Path("chunk.wav"),
        )
        assert chunk.duration_ms == 0


@pytest.mark.unit
class TestChunkPlan:
    """Tests for ChunkPlan model."""

    def test_cleanup_removes_workspace(self, tmp_path: Path) -> None:
        """ChunkPlan.cleanup should remove the workspace directory."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "test.wav").write_bytes(b"audio")

        plan = ChunkPlan(duration_ms=10000, chunks=[], workspace=workspace)
        plan.cleanup()

        assert not workspace.exists()

    def test_cleanup_handles_missing_workspace(self, tmp_path: Path) -> None:
        """ChunkPlan.cleanup should not raise if workspace doesn't exist."""
        workspace = tmp_path / "nonexistent"
        plan = ChunkPlan(duration_ms=10000, chunks=[], workspace=workspace)

        # Should not raise
        plan.cleanup()


@pytest.mark.unit
class TestCalculateCutPoints:
    """Tests for _calculate_cut_points function."""

    def test_single_chunk_for_short_audio(self) -> None:
        """Should return single cut point for audio shorter than max_chunk_ms."""
        # 5 minutes audio, max chunk 10 minutes
        duration_ms = 5 * 60 * 1000
        max_chunk_ms = 10 * 60 * 1000
        min_chunk_ms = 5 * 60 * 1000

        cuts = _calculate_cut_points(duration_ms, [], min_chunk_ms, max_chunk_ms)

        assert len(cuts) == 1
        assert cuts[0] == duration_ms

    def test_multiple_chunks_for_long_audio(self) -> None:
        """Should return multiple cut points for audio longer than max_chunk_ms."""
        # 25 minutes audio, max chunk 10 minutes
        duration_ms = 25 * 60 * 1000
        max_chunk_ms = 10 * 60 * 1000
        min_chunk_ms = 5 * 60 * 1000

        cuts = _calculate_cut_points(duration_ms, [], min_chunk_ms, max_chunk_ms)

        assert len(cuts) >= 2
        # Last cut should be at or before duration
        assert cuts[-1] <= duration_ms

    def test_uses_silence_for_cut_points(self) -> None:
        """Should prefer silence ranges for cut points."""
        duration_ms = 20 * 60 * 1000  # 20 minutes
        max_chunk_ms = 12 * 60 * 1000  # 12 minutes max
        min_chunk_ms = 8 * 60 * 1000  # 8 minutes min

        # Silence at 10 minutes
        silence_at_10min = 10 * 60 * 1000
        silence_ranges = [(silence_at_10min - 500, silence_at_10min + 500)]

        cuts = _calculate_cut_points(
            duration_ms, silence_ranges, min_chunk_ms, max_chunk_ms
        )

        # Should cut near the silence point
        assert any(abs(cut - silence_at_10min) <= 1000 for cut in cuts)

    def test_cut_points_are_monotonically_increasing(self) -> None:
        """Cut points should be in ascending order."""
        duration_ms = 30 * 60 * 1000
        max_chunk_ms = 10 * 60 * 1000
        min_chunk_ms = 5 * 60 * 1000

        cuts = _calculate_cut_points(duration_ms, [], min_chunk_ms, max_chunk_ms)

        for i in range(len(cuts) - 1):
            assert cuts[i] < cuts[i + 1]


@pytest.mark.unit
class TestFindSilenceBetween:
    """Tests for _find_silence_between function."""

    def test_finds_silence_in_window(self) -> None:
        """Should find silence midpoint within the given window."""
        silence_ranges = [(5000, 6000)]  # Silence from 5s to 6s
        result = _find_silence_between(silence_ranges, 4000, 8000)

        assert result == 5500  # Midpoint of 5000-6000

    def test_returns_none_when_no_silence_in_window(self) -> None:
        """Should return None when no silence is within the window."""
        silence_ranges = [(1000, 2000)]  # Silence before window
        result = _find_silence_between(silence_ranges, 5000, 10000)

        assert result is None

    def test_returns_none_for_empty_silence_ranges(self) -> None:
        """Should return None for empty silence ranges."""
        result = _find_silence_between([], 0, 10000)

        assert result is None

    def test_skips_silence_before_window(self) -> None:
        """Should skip silence ranges that end before the window starts."""
        silence_ranges = [(1000, 2000), (7000, 8000)]
        result = _find_silence_between(silence_ranges, 5000, 10000)

        assert result == 7500  # Midpoint of 7000-8000


@pytest.mark.unit
class TestGuessExtension:
    """Tests for _guess_extension function."""

    def test_extracts_mp3_extension(self) -> None:
        """Should extract .mp3 extension from URL."""
        result = _guess_extension("https://example.com/audio.mp3")
        assert result == ".mp3"

    def test_extracts_wav_extension(self) -> None:
        """Should extract .wav extension from URL."""
        result = _guess_extension("https://example.com/audio.wav")
        assert result == ".wav"

    def test_returns_audio_for_no_extension(self) -> None:
        """Should return .audio when URL has no extension."""
        result = _guess_extension("https://example.com/audio")
        assert result == ".audio"

    def test_handles_url_with_query_params(self) -> None:
        """Should handle URLs with query parameters."""
        result = _guess_extension("https://example.com/audio.mp3?token=abc")
        assert result == ".mp3"
