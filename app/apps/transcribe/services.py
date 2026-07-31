"""Transcription task processing services."""

import asyncio
import logging
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path

from beanie.exceptions import CollectionWasNotInitialized
from fastapi_mongo_base.tasks import TaskStatusEnum
from soniox import SonioxClient
from soniox.languages import Language
from soniox.types import (
    TranscriptionConfig,
    TranscriptionJob,
    TranscriptionJobStatus,
    TranscriptionResult,
    TranscriptionWebhook,
)

from server.config import Settings
from utils import conditions, texttools
from utils.billing import finance

from . import chunker_ffmpeg as chunker
from .models import TranscribeTask

CHUNK_STORAGE_ROOT = Path(Settings.storage_path) / "transcribe-chunks"
_soniox_client: SonioxClient | None = None


class SonioxConfigurationError(RuntimeError):
    """Raised when Soniox is used without its required configuration."""


class TranscriptionJobError(RuntimeError):
    """Raised when a Soniox transcription job fails."""


class LazySonioxClient:
    """Patchable lazy proxy for Soniox API methods."""

    def _client(self) -> SonioxClient:
        global _soniox_client
        if _soniox_client is None:
            if not Settings.soniox_api_key:
                error = SonioxConfigurationError("SONIOX_API_KEY is not configured")
                raise error
            _soniox_client = SonioxClient(Settings.soniox_api_key)
        return _soniox_client

    async def transcribe_url_async(
        self,
        url: str,
        config: TranscriptionConfig | None = None,
        **kwargs: object,
    ) -> TranscriptionJob:
        """Proxy transcribe URL calls to a lazily-created client."""
        return await self._client().transcribe_url_async(url, config, **kwargs)

    async def transcribe_file_async(
        self,
        file_path: str,
        config: TranscriptionConfig | None = None,
        **kwargs: object,
    ) -> TranscriptionJob:
        """Proxy transcribe file calls to a lazily-created client."""
        return await self._client().transcribe_file_async(file_path, config, **kwargs)

    async def get_transcription_job_async(
        self,
        job_id: str,
    ) -> TranscriptionJob:
        """Proxy job lookup calls to a lazily-created client."""
        return await self._client().get_transcription_job_async(job_id)

    async def get_transcription_result_async(
        self,
        job_id: str,
    ) -> TranscriptionResult:
        """Proxy result lookup calls to a lazily-created client."""
        return await self._client().get_transcription_result_async(job_id)


soniox = LazySonioxClient()


def get_soniox_client() -> LazySonioxClient:
    """Create a Soniox client only when transcription processing is requested."""
    return soniox


def _task_provider(task: TranscribeTask) -> str:
    """Return a valid provider value from persisted tasks or loose test doubles."""
    provider = getattr(task, "provider", "soniox")
    return provider if isinstance(provider, str) else "soniox"


async def process_transcribe(
    task: TranscribeTask,
    *,
    force_restart: bool = False,
    sync: bool = False,
    **kwargs: object,
) -> TranscribeTask:
    """
    Process a transcription task.

    Chunk audio and submit to transcription service.
    """
    logging.info("Starting processing for task %s", task.uid)
    provider = _task_provider(task)
    if provider != "soniox":
        return await save_error(
            task,
            f"Unsupported transcribe provider: {provider}",
        )

    estimated_cost = finance.estimate_transcribe_cost(
        minutes=task.audio_duration / 60,
        provider=provider,
    )
    quota = await finance.check_quota(
        task.user_id,
        estimated_cost,
        raise_exception=False,
        workspace_id=task.workspace_id,
    )
    if quota < estimated_cost:
        return await save_error(task, "insufficient_quota")

    if Settings.transcribe_enable_chunking:
        logging.info("Chunked transcription enabled for task %s", task.uid)
        try:
            return await _process_chunked_transcribe(task, sync=sync)
        except Exception:
            logging.exception("Chunked transcription failed for %s", task.uid)
            if not Settings.transcribe_chunking_fallback_single:
                return await save_error(task, "chunk_transcription_failed")
            logging.info("Falling back to single job transcription for %s", task.uid)

    return await _process_single_job(task, sync=sync)


async def _process_single_job(task: TranscribeTask, *, sync: bool) -> TranscribeTask:
    soniox = get_soniox_client()
    config = _build_transcription_config(task, chunk_id=None, use_webhook=True)
    if task.file_url.startswith("data:"):
        file_content = await task.file_content()
        suffix = chunker._guess_extension(task.file_url)
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp_file:
            tmp_file.write(file_content.getvalue())
            tmp_file.flush()
            job = await soniox.transcribe_file_async(tmp_file.name, config)
    else:
        job = await soniox.transcribe_url_async(task.file_url, config)

    task.transcription_job_id = job.id
    task.task_status = TaskStatusEnum.processing
    await task.save()
    if not sync:
        return task

    await conditions.Conditions().wait_condition(task.uid)

    try:
        finished_task = await TranscribeTask.get_item(
            task.uid,
            user_id=task.user_id,
            tenant_id=task.tenant_id,
        )
    except CollectionWasNotInitialized:
        finished_task = task
    if not finished_task or not finished_task.transcription_job_id:
        return await save_error(task, "transcription_failed")
    job_result = await soniox.get_transcription_job_async(
        finished_task.transcription_job_id
    )

    if job_result.status != TranscriptionJobStatus.COMPLETED:
        return await save_error(task, "transcription_failed")

    return await process_transcription_webhook(
        finished_task,
        TranscriptionWebhook(
            id=job_result.id,
            status=job_result.status,
        ),
    )


async def _process_chunked_transcribe(
    task: TranscribeTask, *, sync: bool
) -> TranscribeTask:
    chunk_plan = await chunker.create_chunk_plan(
        task_uid=task.uid,
        file_url=task.file_url,
        storage_root=CHUNK_STORAGE_ROOT,
        min_chunk_ms=Settings.transcribe_chunk_min_minutes * 60 * 1000,
        max_chunk_ms=Settings.transcribe_chunk_max_minutes * 60 * 1000,
        silence_len_ms=Settings.transcribe_chunk_min_silence_ms,
        silence_threshold_db=Settings.transcribe_chunk_silence_threshold,
        chunk_format=Settings.transcribe_chunk_format,
    )
    if sync:
        logging.debug("Running chunked transcription synchronously for %s", task.uid)
    logging.info(
        "Task %s chunked into %s segments",
        task.uid,
        len(chunk_plan.chunks),
    )
    task.chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "start_ms": chunk.start_ms,
            "end_ms": chunk.end_ms,
            "file_path": str(chunk.file_path),
        }
        for chunk in chunk_plan.chunks
    ]
    await task.update_and_emit(
        task_report=f"Chunked into {len(chunk_plan.chunks)} segments",
    )
    await task.save()

    try:
        chunk_results = await _transcribe_chunks(task, chunk_plan)
    finally:
        chunk_plan.cleanup()

    ordered_results = sorted(
        chunk_results,
        key=lambda result: (result.chunk.start_ms, result.chunk.chunk_id),
    )
    task.transcription_job_id = (
        ordered_results[0].job_id if ordered_results else task.transcription_job_id
    )
    task.chunks = [
        {
            "chunk_id": result.chunk.chunk_id,
            "start_ms": result.chunk.start_ms,
            "end_ms": result.chunk.end_ms,
            "file_path": str(result.chunk.file_path),
            "job_id": result.job_id,
            "text": result.text,
        }
        for result in ordered_results
    ]
    await task.update_and_emit(
        task_report=f"Transcribed {len(ordered_results)} chunks",
    )
    await task.save()

    combined_text = _combine_chunk_texts(ordered_results)
    total_cost = sum(result.transcription_cost for result in ordered_results)
    usage = None
    try:
        usage = await finance.meter_cost(
            task.user_id,
            total_cost,
            meta_data={
                "service": "transcribe",
                "provider": _task_provider(task),
                "chunks": len(ordered_results),
            },
            workspace_id=task.workspace_id,
        )
    except Exception:
        logging.exception("Failed to meter chunked transcribe usage for %s", task.uid)
    await conditions.Conditions().release_condition(task.uid)
    return await save_result(
        task,
        combined_text,
        float(usage.amount) if usage else total_cost,
        usage.uid if usage else None,
    )


async def _transcribe_chunks(
    task: TranscribeTask,
    chunk_plan: chunker.ChunkPlan,
) -> list[chunker.ChunkTranscriptionResult]:
    if not chunk_plan.chunks:
        return []

    parallelism = max(1, Settings.transcribe_max_parallel_requests)
    semaphore = asyncio.Semaphore(parallelism)

    async def run_chunk(
        audio_chunk: chunker.AudioChunk,
    ) -> chunker.ChunkTranscriptionResult:
        async with semaphore:
            soniox = get_soniox_client()
            job = await soniox.transcribe_file_async(
                str(audio_chunk.file_path),
                _build_transcription_config(task, chunk_id=audio_chunk.chunk_id),
            )
            job_result = await _wait_for_job_completion(job.id)
            if job_result.status != TranscriptionJobStatus.COMPLETED:
                error = TranscriptionJobError(
                    f"Chunk {audio_chunk.chunk_id} "
                    f"failed with status {job_result.status}"
                )
                raise error
            transcript = await soniox.get_transcription_result_async(job.id)
            transcription_cost = math.ceil(
                ((job_result.audio_duration_ms or audio_chunk.duration_ms) / 60000)
                * Settings.minutes_price
            )
            return chunker.ChunkTranscriptionResult(
                chunk=audio_chunk,
                job_id=job.id,
                text=transcript.text,
                audio_duration_ms=(
                    job_result.audio_duration_ms or audio_chunk.duration_ms
                ),
                transcription_cost=transcription_cost,
            )

    tasks = [asyncio.create_task(run_chunk(chunk)) for chunk in chunk_plan.chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [result for result in results if isinstance(result, Exception)]
    if errors:
        for error in errors:
            logging.exception("Chunk transcription error: %s", error)
        raise errors[0]

    return [
        result
        for result in results
        if isinstance(result, chunker.ChunkTranscriptionResult)
    ]


async def _wait_for_job_completion(job_id: str) -> TranscriptionJob:
    soniox = get_soniox_client()
    while True:
        job = await soniox.get_transcription_job_async(job_id)
        if job.status == TranscriptionJobStatus.COMPLETED:
            return job
        if job.status == TranscriptionJobStatus.ERROR:
            error = TranscriptionJobError(f"Job {job_id} failed: {job.error_message}")
            raise error
        await asyncio.sleep(Settings.transcribe_poll_interval_seconds)


def _combine_chunk_texts(results: Sequence[chunker.ChunkTranscriptionResult]) -> str:
    ordered = sorted(
        results, key=lambda result: (result.chunk.start_ms, result.chunk.chunk_id)
    )
    parts = []
    for result in ordered:
        text = (result.text or "").strip()
        if text:
            parts.append(text)
    if not parts:
        return ""
    combined = "\n\n".join(parts)
    return texttools.normalize_text(combined)


def _build_transcription_config(
    task: TranscribeTask,
    *,
    chunk_id: int | None,
    use_webhook: bool = False,
) -> TranscriptionConfig:
    client_reference = f"{task.uid}:{chunk_id}" if chunk_id is not None else task.uid
    webhook_url = None
    if use_webhook:
        suffix = f"transcribes/{task.uid}/webhook"
        if chunk_id is not None:
            suffix = f"{suffix}/{chunk_id}"
        webhook_url = f"https://{Settings.root_url}{Settings.base_path}/{suffix}"
        from .webhook_auth import append_webhook_auth

        webhook_url = append_webhook_auth(webhook_url, task.uid)
    return TranscriptionConfig(
        language_hints=[Language.PERSIAN, Language.ENGLISH],
        enable_language_identification=True,
        enable_speaker_diarization=True,
        client_reference_id=client_reference,
        webhook_url=webhook_url,
    )


async def save_error(
    task: TranscribeTask, message: str, **kwargs: object
) -> TranscribeTask:
    """Save error result for a transcription task."""
    task.task_status = TaskStatusEnum.error
    await task.update_and_emit(
        task_report=message, log_type=kwargs.get("log_type", "error")
    )
    await conditions.Conditions().release_condition(task.uid)
    logging.warning("Transcription rejected %s", f"{message}\n\n{kwargs}")
    return task


async def save_result(
    task: TranscribeTask,
    result: str,
    usage_amount: float | None = None,
    usage_id: str | None = None,
) -> TranscribeTask:
    """Save successful result for a transcription task."""
    task.result = texttools.normalize_text(result)
    task.task_status = TaskStatusEnum.completed
    task.usage_amount = usage_amount
    task.usage_id = usage_id
    task.provider_meta = {
        "provider": task.provider,
        "model": task.model,
        "usage": {"audio_duration_seconds": task.audio_duration},
    }
    await task.update_and_emit(task_report="Task processed successfully")
    return task


async def process_transcription_webhook(
    task: TranscribeTask,
    data: TranscriptionWebhook,
) -> TranscribeTask:
    """Process transcription completion webhook and save results."""
    translation_cost = 0

    if not task.transcription_job_id or task.transcription_job_id != data.id:
        return await process_error_webhook(task, "Transcription job ID does not match")
    if data.status != TranscriptionJobStatus.COMPLETED:
        return await process_error_webhook(task, "Transcription job status is error")
    if data.status == TranscriptionJobStatus.ERROR:
        return await process_error_webhook(task, "Transcription job status is error")

    soniox = get_soniox_client()
    job_result = await soniox.get_transcription_job_async(task.transcription_job_id)

    transcription_cost = finance.estimate_transcribe_cost(
        minutes=(job_result.audio_duration_ms or 0) / 60 / 1000,
        provider=task.provider,
    )
    total_cost = transcription_cost + translation_cost
    # The transcription already happened (Soniox was already paid) by the
    # time this runs -- a billing-recording failure (transient outage, or
    # actual usage running slightly over the pre-flight estimate) must be
    # logged, not allowed to withhold a result that's already been produced.
    usage = None
    try:
        usage = await finance.meter_cost(
            task.user_id,
            total_cost,
            meta_data={
                "service": "transcribe",
                "provider": task.provider,
                "model": task.model,
                "job_id": task.transcription_job_id,
                "audio_duration_ms": job_result.audio_duration_ms,
            },
            workspace_id=task.workspace_id,
        )
    except Exception:
        logging.exception("Failed to meter transcribe usage for task %s", task.uid)
    logging.info(
        "%s %s %s %s",
        task.uid,
        job_result.audio_duration_ms,
        total_cost,
        transcription_cost,
    )

    result = await soniox.get_transcription_result_async(task.transcription_job_id)

    await conditions.Conditions().release_condition(task.uid)
    return await save_result(
        task,
        result.text,
        float(usage.amount) if usage else transcription_cost,
        usage.uid if usage else None,
    )


async def process_error_webhook(
    task: TranscribeTask, message: str = ""
) -> TranscribeTask:
    """Process error webhook for a failed transcription task."""
    if not task.transcription_job_id:
        return await save_error(task, "Transcription job ID is required")
    soniox = get_soniox_client()
    job = await soniox.get_transcription_job_async(task.transcription_job_id)

    return await save_error(task, message, error_message=job.error_message)
