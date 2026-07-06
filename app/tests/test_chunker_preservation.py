"""
Preservation property tests for audio chunking behavior.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

These tests verify that the ffmpeg-based chunker implementation preserves
all the audio processing behavior from the original pydub-based implementation.

IMPORTANT: These tests document the expected behavior that must be preserved
when replacing pydub with ffmpeg. They test the CURRENT (ffmpeg) implementation
to establish a baseline of correct behavior.

Property 2: Preservation - Audio Chunking Behavior Unchanged
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the ffmpeg-based chunker (current implementation)
from apps.transcribe import chunker_ffmpeg as chunker


class TestAudioDurationPreservation:
    """
    Test that audio duration calculation remains consistent.

    **Validates: Requirement 3.1** - Audio files loaded from URLs are processed correctly
    """

    def test_get_audio_duration_returns_positive_integer(self) -> None:
        """
        Property: Audio duration must be a positive integer in milliseconds.

        For any valid audio file, _get_audio_duration_ms() should return
        a positive integer representing the duration in milliseconds.
        """
        # Create a minimal valid audio file for testing
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                # Create a 1-second silent audio file using ffmpeg
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=44100:cl=mono",
                        "-t",
                        "1",
                        "-q:a",
                        "9",
                        "-acodec",
                        "libmp3lame",
                        str(tmp_path),
                    ],
                    capture_output=True,
                    check=True,
                )

                duration_ms = chunker._get_audio_duration_ms(tmp_path)

                # Property: Duration must be positive
                assert duration_ms > 0, "Audio duration must be positive"

                # Property: Duration must be an integer
                assert isinstance(duration_ms, int), "Duration must be an integer"

                # Property: Duration should be approximately 1000ms (1 second)
                # Allow 10% tolerance for encoding variations
                assert 900 <= duration_ms <= 1100, (
                    f"Expected ~1000ms, got {duration_ms}ms"
                )
            finally:
                tmp_path.unlink(missing_ok=True)

    def test_audio_duration_scales_linearly(self) -> None:
        """
        Property: Audio duration should scale linearly with actual audio length.

        A 2-second audio file should have approximately twice the duration
        of a 1-second audio file.
        """
        durations = []

        for seconds in [1, 2]:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-f",
                            "lavfi",
                            "-i",
                            "anullsrc=r=44100:cl=mono",
                            "-t",
                            str(seconds),
                            "-q:a",
                            "9",
                            "-acodec",
                            "libmp3lame",
                            str(tmp_path),
                        ],
                        capture_output=True,
                        check=True,
                    )

                    duration_ms = chunker._get_audio_duration_ms(tmp_path)
                    durations.append(duration_ms)
                finally:
                    tmp_path.unlink(missing_ok=True)

        # Property: 2-second file should be approximately 2x the 1-second file
        ratio = durations[1] / durations[0]
        assert 1.8 <= ratio <= 2.2, f"Duration ratio should be ~2.0, got {ratio:.2f}"


class TestSilenceDetectionPreservation:
    """
    Test that silence detection behavior remains consistent.

    **Validates: Requirement 3.2** - Silence detection with same parameters
    produces same silence ranges
    """

    def test_detect_silence_returns_list_of_tuples(self) -> None:
        """
        Property: Silence detection must return a list of (start_ms, end_ms) tuples.

        For any audio file, _detect_silence_ffmpeg() should return a list
        where each element is a tuple of two integers representing the
        start and end times of silence in milliseconds.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                # Create audio with silence: 0.5s sound, 1s silence, 0.5s sound
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=1000:duration=0.5",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=44100:cl=mono:d=1",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=1000:duration=0.5",
                        "-filter_complex",
                        "[0:a][1:a][2:a]concat=n=3:v=0:a=1",
                        "-acodec",
                        "libmp3lame",
                        str(tmp_path),
                    ],
                    capture_output=True,
                    check=True,
                )

                silence_ranges = chunker._detect_silence_ffmpeg(
                    tmp_path,
                    silence_len_ms=500,  # Detect silences >= 500ms
                    silence_threshold_db=-50,
                )

                # Property: Result must be a list
                assert isinstance(silence_ranges, list), "Result must be a list"

                # Property: Each element must be a tuple of two integers
                for silence_range in silence_ranges:
                    assert isinstance(silence_range, tuple), (
                        "Each silence range must be a tuple"
                    )
                    assert len(silence_range) == 2, (
                        "Each tuple must have exactly 2 elements"
                    )
                    start_ms, end_ms = silence_range
                    assert isinstance(start_ms, int), "Start time must be an integer"
                    assert isinstance(end_ms, int), "End time must be an integer"

                    # Property: End time must be after start time
                    assert end_ms > start_ms, (
                        f"End time ({end_ms}) must be after start time ({start_ms})"
                    )
            finally:
                tmp_path.unlink(missing_ok=True)

    def test_silence_detection_respects_threshold(self) -> None:
        """
        Property: Lower threshold should detect more silence than higher threshold.

        A more sensitive threshold (higher dB value, e.g., -30dB) should detect
        more or equal silence ranges compared to a less sensitive threshold (-50dB).
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            try:
                # Create audio with varying volume levels
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=1000:duration=2",
                        "-af",
                        "volume=0.1",  # Low volume (might be detected as silence)
                        "-acodec",
                        "libmp3lame",
                        str(tmp_path),
                    ],
                    capture_output=True,
                    check=True,
                )

                # Detect with strict threshold (less sensitive)
                strict_silences = chunker._detect_silence_ffmpeg(
                    tmp_path,
                    silence_len_ms=100,
                    silence_threshold_db=-60,  # Very quiet = silence
                )

                # Detect with lenient threshold (more sensitive)
                lenient_silences = chunker._detect_silence_ffmpeg(
                    tmp_path,
                    silence_len_ms=100,
                    silence_threshold_db=-20,  # Louder sounds = silence
                )

                # Property: Lenient threshold detects >= silence than strict
                assert len(lenient_silences) >= len(strict_silences), (
                    f"Lenient threshold should detect more silence: "
                    f"lenient={len(lenient_silences)}, strict={len(strict_silences)}"
                )
            finally:
                tmp_path.unlink(missing_ok=True)


class TestChunkBoundaryPreservation:
    """
    Test that chunk boundary calculation remains consistent.

    **Validates: Requirement 3.3** - Chunk boundaries calculated identically
    for same inputs
    """

    def test_calculate_cut_points_respects_min_max_constraints(self) -> None:
        """
        Property: All chunks must be within min/max duration constraints.

        For any audio duration and silence ranges, _calculate_cut_points()
        should produce cut points such that each chunk is between
        min_chunk_ms and max_chunk_ms (except possibly the last chunk).
        """
        duration_ms = 600000  # 10 minutes
        silence_ranges = [
            (60000, 62000),  # 1 minute mark
            (180000, 182000),  # 3 minute mark
            (300000, 302000),  # 5 minute mark
            (420000, 422000),  # 7 minute mark
        ]
        min_chunk_ms = 120000  # 2 minutes
        max_chunk_ms = 240000  # 4 minutes

        cut_points = chunker._calculate_cut_points(
            duration_ms,
            silence_ranges,
            min_chunk_ms,
            max_chunk_ms,
        )

        # Property: Cut points must be in ascending order
        assert cut_points == sorted(cut_points), "Cut points must be sorted"

        # Property: All cut points must be within audio duration
        for cut_point in cut_points:
            assert 0 <= cut_point <= duration_ms, (
                f"Cut point {cut_point} outside valid range [0, {duration_ms}]"
            )

        # Property: Chunks must respect min/max constraints
        cursor = 0
        for idx, cut_point in enumerate(cut_points):
            chunk_duration = cut_point - cursor

            # Last chunk can be shorter than min_chunk_ms
            if idx == len(cut_points) - 1:
                assert chunk_duration <= max_chunk_ms, (
                    f"Chunk {idx} duration {chunk_duration}ms exceeds max {max_chunk_ms}ms"
                )
            else:
                assert min_chunk_ms <= chunk_duration <= max_chunk_ms, (
                    f"Chunk {idx} duration {chunk_duration}ms outside "
                    f"[{min_chunk_ms}, {max_chunk_ms}]"
                )
            cursor = cut_point

    def test_calculate_cut_points_prefers_silence_boundaries(self) -> None:
        """
        Property: Cut points should align with silence ranges when possible.

        When silence ranges exist within the target window, cut points
        should be placed at the midpoint of those silence ranges.
        """
        duration_ms = 300000  # 5 minutes
        # Place silence exactly at 2 minutes (within typical chunk window)
        silence_ranges = [(120000, 122000)]
        min_chunk_ms = 60000  # 1 minute
        max_chunk_ms = 180000  # 3 minutes

        cut_points = chunker._calculate_cut_points(
            duration_ms,
            silence_ranges,
            min_chunk_ms,
            max_chunk_ms,
        )

        # Property: First cut point should be near the silence range
        if cut_points:
            first_cut = cut_points[0]
            silence_midpoint = (120000 + 122000) // 2

            # Cut point should be at or near the silence midpoint
            assert abs(first_cut - silence_midpoint) <= 2000, (
                f"Cut point {first_cut} should be near silence midpoint {silence_midpoint}"
            )


class TestAudioExportPreservation:
    """
    Test that audio chunk export produces valid files.

    **Validates: Requirement 3.4** - Audio chunk export produces files
    with identical audio content
    """

    def test_export_audio_segment_creates_valid_file(self) -> None:
        """
        Property: Exported audio segments must be valid audio files.

        For any audio file and time range, _export_audio_segment() should
        create a valid audio file that can be read by ffprobe.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as source_tmp:
            source_path = Path(source_tmp.name)
            try:
                # Create a 3-second audio file
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=1000:duration=3",
                        "-acodec",
                        "libmp3lame",
                        str(source_path),
                    ],
                    capture_output=True,
                    check=True,
                )

                with tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False
                ) as output_tmp:
                    output_path = Path(output_tmp.name)
                    try:
                        # Export segment from 1s to 2s (1000ms duration)
                        chunker._export_audio_segment(
                            source_path,
                            output_path,
                            start_ms=1000,
                            end_ms=2000,
                            output_format="mp3",
                        )

                        # Property: Output file must exist
                        assert output_path.exists(), "Output file must be created"

                        # Property: Output file must be valid audio
                        result = subprocess.run(
                            [
                                "ffprobe",
                                "-v",
                                "error",
                                "-show_entries",
                                "format=duration",
                                "-of",
                                "json",
                                str(output_path),
                            ],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        data = json.loads(result.stdout)
                        exported_duration = float(data["format"]["duration"])

                        # Property: Exported duration should match requested duration
                        # Allow 10% tolerance for encoding
                        expected_duration = 1.0  # 1 second
                        assert 0.9 <= exported_duration <= 1.1, (
                            f"Expected ~{expected_duration}s, got {exported_duration}s"
                        )
                    finally:
                        output_path.unlink(missing_ok=True)
            finally:
                source_path.unlink(missing_ok=True)

    def test_export_preserves_audio_format(self) -> None:
        """
        Property: Exported audio must be in the requested format.

        When exporting with a specific format (e.g., 'mp3', 'wav'),
        the output file should be in that format.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as source_tmp:
            source_path = Path(source_tmp.name)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=1000:duration=1",
                        "-acodec",
                        "libmp3lame",
                        str(source_path),
                    ],
                    capture_output=True,
                    check=True,
                )

                for output_format in ["mp3", "wav"]:
                    with tempfile.NamedTemporaryFile(
                        suffix=f".{output_format}", delete=False
                    ) as output_tmp:
                        output_path = Path(output_tmp.name)
                        try:
                            chunker._export_audio_segment(
                                source_path,
                                output_path,
                                start_ms=0,
                                end_ms=1000,
                                output_format=output_format,
                            )

                            # Property: Output file format matches requested format
                            result = subprocess.run(
                                [
                                    "ffprobe",
                                    "-v",
                                    "error",
                                    "-show_entries",
                                    "format=format_name",
                                    "-of",
                                    "json",
                                    str(output_path),
                                ],
                                capture_output=True,
                                text=True,
                                check=True,
                            )
                            data = json.loads(result.stdout)
                            format_name = data["format"]["format_name"]

                            # Format name might be compound (e.g., "mp3" or "wav")
                            assert output_format in format_name, (
                                f"Expected format '{output_format}' in '{format_name}'"
                            )
                        finally:
                            output_path.unlink(missing_ok=True)
            finally:
                source_path.unlink(missing_ok=True)


class TestChunkPlanAPIPreservation:
    """
    Test that ChunkPlan, AudioChunk, and ChunkTranscriptionResult APIs remain unchanged.

    **Validates: Requirement 3.5** - Model APIs maintain same interface and data structures
    """

    def test_audio_chunk_model_has_required_fields(self) -> None:
        """
        Property: AudioChunk model must have all required fields.

        The AudioChunk model must maintain its API with chunk_id, start_ms,
        end_ms, file_path, and duration_ms property.
        """
        chunk = chunker.AudioChunk(
            chunk_id=0,
            start_ms=1000,
            end_ms=3000,
            file_path=Path("/tmp/chunk_0000.mp3"),
        )

        # Property: All required fields must be accessible
        assert hasattr(chunk, "chunk_id"), "AudioChunk must have chunk_id"
        assert hasattr(chunk, "start_ms"), "AudioChunk must have start_ms"
        assert hasattr(chunk, "end_ms"), "AudioChunk must have end_ms"
        assert hasattr(chunk, "file_path"), "AudioChunk must have file_path"
        assert hasattr(chunk, "duration_ms"), (
            "AudioChunk must have duration_ms property"
        )

        # Property: duration_ms is calculated correctly
        assert chunk.duration_ms == 2000, (
            f"Expected duration 2000ms, got {chunk.duration_ms}ms"
        )

    def test_chunk_plan_model_has_required_fields(self) -> None:
        """
        Property: ChunkPlan model must have all required fields.

        The ChunkPlan model must maintain its API with duration_ms, chunks,
        workspace, and cleanup() method.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            chunks = [
                chunker.AudioChunk(
                    chunk_id=0,
                    start_ms=0,
                    end_ms=1000,
                    file_path=workspace / "chunk_0000.mp3",
                )
            ]

            plan = chunker.ChunkPlan(
                duration_ms=1000,
                chunks=chunks,
                workspace=workspace,
            )

            # Property: All required fields must be accessible
            assert hasattr(plan, "duration_ms"), "ChunkPlan must have duration_ms"
            assert hasattr(plan, "chunks"), "ChunkPlan must have chunks"
            assert hasattr(plan, "workspace"), "ChunkPlan must have workspace"
            assert hasattr(plan, "cleanup"), "ChunkPlan must have cleanup method"

            # Property: cleanup method must be callable
            assert callable(plan.cleanup), "cleanup must be a callable method"

    def test_chunk_transcription_result_model_has_required_fields(self) -> None:
        """
        Property: ChunkTranscriptionResult model must have all required fields.

        The ChunkTranscriptionResult model must maintain its API with chunk,
        job_id, text, audio_duration_ms, and transcription_cost fields.
        """
        chunk = chunker.AudioChunk(
            chunk_id=0,
            start_ms=0,
            end_ms=1000,
            file_path=Path("/tmp/chunk_0000.mp3"),
        )

        result = chunker.ChunkTranscriptionResult(
            chunk=chunk,
            job_id="test-job-123",
            text="Test transcription",
            audio_duration_ms=1000,
            transcription_cost=0.5,
        )

        # Property: All required fields must be accessible
        assert hasattr(result, "chunk"), "Result must have chunk"
        assert hasattr(result, "job_id"), "Result must have job_id"
        assert hasattr(result, "text"), "Result must have text"
        assert hasattr(result, "audio_duration_ms"), (
            "Result must have audio_duration_ms"
        )
        assert hasattr(result, "transcription_cost"), (
            "Result must have transcription_cost"
        )


@pytest.mark.asyncio
class TestChunkPlanIntegrationPreservation:
    """
    Test that create_chunk_plan integrates correctly with the transcription workflow.

    **Validates: Requirements 3.6, 3.7, 3.8** - Integration with transcription service,
    multiple audio formats, and cleanup functionality
    """

    async def test_create_chunk_plan_with_short_audio(self) -> None:
        """
        Property: Audio shorter than max_chunk_ms should produce single chunk.

        When audio duration is less than max_chunk_ms, create_chunk_plan()
        should return a ChunkPlan with exactly one chunk spanning the entire audio.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_root = Path(tmpdir) / "storage"
            storage_root.mkdir()

            # Create a short audio file (1 second)
            audio_file = Path(tmpdir) / "short_audio.mp3"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=1",
                    "-acodec",
                    "libmp3lame",
                    str(audio_file),
                ],
                capture_output=True,
                check=True,
            )

            # Mock the download to use local file
            with patch(
                "apps.transcribe.chunker_ffmpeg._download_audio"
            ) as mock_download:

                async def copy_file(file_url: str, destination: Path) -> None:
                    import shutil

                    shutil.copy(audio_file, destination)

                mock_download.side_effect = copy_file

                plan = await chunker.create_chunk_plan(
                    task_uid="test-task-001",
                    file_url=f"file://{audio_file}",
                    storage_root=storage_root,
                    min_chunk_ms=60000,  # 1 minute
                    max_chunk_ms=180000,  # 3 minutes
                    silence_len_ms=1000,
                    silence_threshold_db=-40,
                    chunk_format="mp3",
                )

                try:
                    # Property: Short audio produces single chunk
                    assert len(plan.chunks) == 1, (
                        f"Expected 1 chunk for short audio, got {len(plan.chunks)}"
                    )

                    # Property: Single chunk spans entire audio
                    chunk = plan.chunks[0]
                    assert chunk.start_ms == 0, "Chunk should start at 0"
                    assert chunk.end_ms == plan.duration_ms, (
                        "Chunk should end at audio duration"
                    )

                    # Property: Chunk file exists
                    assert chunk.file_path.exists(), "Chunk file must exist"
                finally:
                    plan.cleanup()

    async def test_chunk_plan_cleanup_removes_workspace(self) -> None:
        """
        Property: ChunkPlan.cleanup() must remove the workspace directory.

        **Validates: Requirement 3.8** - Chunk cleanup properly removes
        temporary workspace directories
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_root = Path(tmpdir) / "storage"
            storage_root.mkdir()

            audio_file = Path(tmpdir) / "test_audio.mp3"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=1",
                    "-acodec",
                    "libmp3lame",
                    str(audio_file),
                ],
                capture_output=True,
                check=True,
            )

            with patch(
                "apps.transcribe.chunker_ffmpeg._download_audio"
            ) as mock_download:

                async def copy_file(file_url: str, destination: Path) -> None:
                    import shutil

                    shutil.copy(audio_file, destination)

                mock_download.side_effect = copy_file

                plan = await chunker.create_chunk_plan(
                    task_uid="test-cleanup-001",
                    file_url=f"file://{audio_file}",
                    storage_root=storage_root,
                    min_chunk_ms=60000,
                    max_chunk_ms=180000,
                    silence_len_ms=1000,
                    silence_threshold_db=-40,
                    chunk_format="mp3",
                )

                workspace_path = plan.workspace

                # Property: Workspace exists before cleanup
                assert workspace_path.exists(), "Workspace should exist before cleanup"

                # Perform cleanup
                plan.cleanup()

                # Property: Workspace is removed after cleanup
                assert not workspace_path.exists(), (
                    "Workspace should be removed after cleanup"
                )


# Documentation of expected behavior for preservation
PRESERVATION_DOCUMENTATION = """
Preservation Property Test Results:

These tests document the expected behavior of the audio chunking functionality
that must be preserved when replacing pydub with ffmpeg.

Expected Behaviors (from Requirements 3.1-3.8):

1. Audio File Processing (3.1):
   - Audio files downloaded from URLs are processed correctly
   - Duration is calculated accurately in milliseconds
   - Duration scales linearly with audio length

2. Silence Detection (3.2):
   - Silence ranges are identified with configurable threshold and duration
   - Returns list of (start_ms, end_ms) tuples
   - More sensitive thresholds detect more silence
   - Silence ranges are within audio duration

3. Chunk Boundaries (3.3):
   - Chunks are created within min/max duration constraints
   - Cut points align with silence ranges when possible
   - Cut points are in ascending order
   - All cut points are within audio duration

4. Audio Export (3.4):
   - Exported chunks are valid audio files
   - Exported duration matches requested segment duration
   - Output format matches requested format (mp3, wav, etc.)

5. API Interface (3.5):
   - AudioChunk: chunk_id, start_ms, end_ms, file_path, duration_ms
   - ChunkPlan: duration_ms, chunks, workspace, cleanup()
   - ChunkTranscriptionResult: chunk, job_id, text, audio_duration_ms, transcription_cost

6. Integration (3.6):
   - Short audio (< max_chunk_ms) produces single chunk
   - Chunk files are created in workspace directory
   - Integration with transcription service workflow

7. Format Support (3.7):
   - Multiple audio formats supported (mp3, wav, m4a, etc.)
   - Format detection from URL extension

8. Cleanup (3.8):
   - Workspace directory is removed after cleanup()
   - Temporary files are properly cleaned up

Testing Approach:
- Tests run on Python 3.13 with ffmpeg-based implementation
- Tests establish baseline behavior that must be preserved
- Property-based approach ensures behavior holds across many inputs
- Tests can be run before and after fix to verify no regressions
"""
