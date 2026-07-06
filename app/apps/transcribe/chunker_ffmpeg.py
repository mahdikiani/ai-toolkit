"""Utilities for splitting audio files into silence-aware chunks using ffmpeg."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import subprocess  # noqa: S404
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)


class AudioChunk(BaseModel):
    """Represents a single chunk of the original audio."""

    chunk_id: int
    start_ms: int
    end_ms: int
    file_path: Path

    @property
    def duration_ms(self) -> int:
        """Calculate chunk duration in milliseconds."""
        return self.end_ms - self.start_ms


class ChunkPlan(BaseModel):
    """Holds the chunking result and manages temporary workspace cleanup."""

    duration_ms: int
    chunks: list[AudioChunk]
    workspace: Path

    def cleanup(self) -> None:
        """Remove temporary workspace directory."""
        shutil.rmtree(self.workspace, ignore_errors=True)


class ChunkTranscriptionResult(BaseModel):
    """Result of transcribing a single audio chunk."""

    chunk: AudioChunk
    job_id: str
    text: str = Field(default="")
    audio_duration_ms: int = Field(default=0)
    transcription_cost: float = Field(default=0)


async def create_chunk_plan(
    *,
    task_uid: str,
    file_url: str,
    storage_root: Path,
    min_chunk_ms: int,
    max_chunk_ms: int,
    silence_len_ms: int,
    silence_threshold_db: int,
    chunk_format: str,
) -> ChunkPlan:
    """Download the audio file, detect silences, and export chunk files."""

    workspace = storage_root / task_uid
    workspace.mkdir(parents=True, exist_ok=True)
    source_path = workspace / f"source{_guess_extension(file_url)}"
    await _download_audio(file_url, source_path)

    duration_ms, chunks = await asyncio.to_thread(
        _run_chunking_pipeline,
        source_path,
        workspace,
        min_chunk_ms,
        max_chunk_ms,
        silence_len_ms,
        silence_threshold_db,
        chunk_format,
    )

    return ChunkPlan(duration_ms=duration_ms, chunks=chunks, workspace=workspace)


async def _download_audio(file_url: str, destination: Path) -> None:
    """Download audio file from URL."""
    async with (
        httpx.AsyncClient(follow_redirects=True) as client,
        client.stream("GET", file_url, timeout=None) as response,
    ):
        response.raise_for_status()
        with destination.open("wb") as file_handle:
            async for chunk in response.aiter_bytes():
                file_handle.write(chunk)


def _run_chunking_pipeline(
    source_path: Path,
    workspace: Path,
    min_chunk_ms: int,
    max_chunk_ms: int,
    silence_len_ms: int,
    silence_threshold_db: int,
    chunk_format: str,
) -> tuple[int, list[AudioChunk]]:
    """Run the complete chunking pipeline using ffmpeg."""
    duration_ms = _get_audio_duration_ms(source_path)

    if duration_ms <= max_chunk_ms:
        chunk_path = workspace / f"chunk_0000.{chunk_format}"
        _export_audio_segment(source_path, chunk_path, 0, duration_ms, chunk_format)
        return duration_ms, [
            AudioChunk(chunk_id=0, start_ms=0, end_ms=duration_ms, file_path=chunk_path)
        ]

    silence_ranges = _detect_silence_ffmpeg(
        source_path,
        silence_len_ms,
        silence_threshold_db,
    )

    cut_points = _calculate_cut_points(
        duration_ms, silence_ranges, min_chunk_ms, max_chunk_ms
    )
    chunks: list[AudioChunk] = []
    cursor = 0

    for idx, cut_point in enumerate(cut_points):
        chunk_path = workspace / f"chunk_{idx:04d}.{chunk_format}"
        _export_audio_segment(source_path, chunk_path, cursor, cut_point, chunk_format)
        chunks.append(
            AudioChunk(
                chunk_id=idx,
                start_ms=cursor,
                end_ms=cut_point,
                file_path=chunk_path,
            )
        )
        cursor = cut_point

    if cursor < duration_ms:
        idx = len(chunks)
        chunk_path = workspace / f"chunk_{idx:04d}.{chunk_format}"
        _export_audio_segment(
            source_path, chunk_path, cursor, duration_ms, chunk_format
        )
        chunks.append(
            AudioChunk(
                chunk_id=idx,
                start_ms=cursor,
                end_ms=duration_ms,
                file_path=chunk_path,
            )
        )

    LOGGER.info("Generated %s chunks for %s", len(chunks), source_path.name)
    return duration_ms, chunks


def _get_audio_duration_ms(audio_path: Path) -> int:
    """Get audio duration in milliseconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
    data = json.loads(result.stdout)
    duration_seconds = float(data["format"]["duration"])
    return int(duration_seconds * 1000)


def _detect_silence_ffmpeg(
    audio_path: Path,
    silence_len_ms: int,
    silence_threshold_db: int,
) -> list[tuple[int, int]]:
    """Detect silence ranges using ffmpeg silencedetect filter."""
    silence_duration = silence_len_ms / 1000.0
    cmd = [
        "ffmpeg",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=noise={silence_threshold_db}dB:d={silence_duration}",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    silence_ranges: list[tuple[int, int]] = []
    lines = result.stderr.split("\n")
    silence_start = None

    for line in lines:
        if "silence_start:" in line:
            parts = line.split("silence_start:")
            if len(parts) > 1:
                try:
                    silence_start = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    continue
        elif "silence_end:" in line and silence_start is not None:
            parts = line.split("silence_end:")
            if len(parts) > 1:
                try:
                    silence_end = float(parts[1].strip().split()[0])
                    silence_ranges.append((
                        int(silence_start * 1000),
                        int(silence_end * 1000),
                    ))
                    silence_start = None
                except (ValueError, IndexError):
                    continue

    return silence_ranges


def _export_audio_segment(
    source_path: Path,
    output_path: Path,
    start_ms: int,
    end_ms: int,
    output_format: str,
) -> None:
    """Export a segment of audio using ffmpeg."""
    start_seconds = start_ms / 1000.0
    duration_seconds = (end_ms - start_ms) / 1000.0

    cmd = [
        "ffmpeg",
        "-i",
        str(source_path),
        "-ss",
        str(start_seconds),
        "-t",
        str(duration_seconds),
        "-c",
        "copy",
        "-y",
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)  # noqa: S603


def _calculate_cut_points(
    duration_ms: int,
    silence_ranges: Sequence[tuple[int, int]],
    min_chunk_ms: int,
    max_chunk_ms: int,
) -> list[int]:
    """Determine cut points so each chunk is within the desired window."""

    normalized_silence = sorted(
        (max(0, start), min(duration_ms, end)) for start, end in silence_ranges
    )
    cuts: list[int] = []
    cursor = 0

    while cursor < duration_ms:
        target_min = min(duration_ms, cursor + min_chunk_ms)
        target_max = min(duration_ms, cursor + max_chunk_ms)
        cut_at = _find_silence_between(normalized_silence, target_min, target_max)
        if cut_at is None:
            cut_at = target_max
        cuts.append(cut_at)
        cursor = cut_at

    return cuts


def _find_silence_between(
    silence_ranges: Sequence[tuple[int, int]],
    window_start: int,
    window_end: int,
) -> int | None:
    """Find the midpoint of a silence range within the given window."""
    for start, end in silence_ranges:
        if end < window_start:
            continue
        if start > window_end:
            break
        midpoint = math.floor((start + end) / 2)
        return min(max(midpoint, window_start), window_end)
    return None


def _guess_extension(file_url: str) -> str:
    """Guess file extension from URL."""
    parsed = urlparse(file_url)
    suffix = Path(parsed.path).suffix
    return suffix or ".audio"
