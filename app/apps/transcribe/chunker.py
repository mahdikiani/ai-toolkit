"""Utilities for splitting audio files into silence-aware chunks."""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field
from pydub import AudioSegment
from pydub.silence import detect_silence

LOGGER = logging.getLogger(__name__)


class AudioChunk(BaseModel):
    """Represents a single chunk of the original audio."""

    chunk_id: int
    start_ms: int
    end_ms: int
    file_path: Path

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class ChunkPlan(BaseModel):
    """Holds the chunking result and manages temporary workspace cleanup."""

    duration_ms: int
    chunks: list[AudioChunk]
    workspace: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)


class ChunkTranscriptionResult(BaseModel):
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
    audio = AudioSegment.from_file(source_path)
    duration_ms = len(audio)
    if duration_ms <= max_chunk_ms:
        chunk_path = workspace / f"chunk_0000.{chunk_format}"
        audio.export(chunk_path, format=chunk_format)
        return duration_ms, [
            AudioChunk(chunk_id=0, start_ms=0, end_ms=duration_ms, file_path=chunk_path)
        ]

    silence_ranges = detect_silence(
        audio,
        min_silence_len=silence_len_ms,
        silence_thresh=silence_threshold_db,
    )

    cut_points = _calculate_cut_points(
        duration_ms, silence_ranges, min_chunk_ms, max_chunk_ms
    )
    chunks: list[AudioChunk] = []
    cursor = 0

    for idx, cut_point in enumerate(cut_points):
        segment = audio[cursor:cut_point]
        chunk_path = workspace / f"chunk_{idx:04d}.{chunk_format}"
        segment.export(chunk_path, format=chunk_format)
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
        audio[cursor:].export(chunk_path, format=chunk_format)
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


def _calculate_cut_points(
    duration_ms: int,
    silence_ranges: Sequence[Sequence[int]],
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
    silence_ranges: Sequence[Sequence[int]],
    window_start: int,
    window_end: int,
) -> int | None:
    for start, end in silence_ranges:
        if end < window_start:
            continue
        if start > window_end:
            break
        return math.floor((start + end) / 2)
    return None


def _guess_extension(file_url: str) -> str:
    parsed = urlparse(file_url)
    suffix = Path(parsed.path).suffix
    return suffix or ".audio"
