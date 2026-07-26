"""Unit tests for transcription services."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.transcribe.schemas import (
    ChunkMetadata,
    TranscribeTaskBase64Schema,
    TranscribeTaskSchema,
)
from apps.transcribe.services import (
    _combine_chunk_texts,
    save_error,
    save_result,
)


def _transcribe_task(**overrides: object) -> TranscribeTaskSchema:
    data: dict[str, object] = {
        "uid": "task_123",
        "user_id": "user_123",
        "tenant_id": "tenant_123",
        "file_url": "https://example.com/audio.mp3",
    }
    data.update(overrides)
    return TranscribeTaskSchema(**data)


@pytest.mark.unit
class TestTranscribeAudioDuration:
    """Tests for TranscribeTaskSchema audio_duration property."""

    def test_uses_explicit_audio_duration_seconds(self) -> None:
        """audio_duration should prefer explicit client-provided seconds."""
        task = _transcribe_task(audio_duration_seconds=12.5)

        assert task.audio_duration == pytest.approx(12.5)

    def test_uses_provider_meta_duration(self) -> None:
        """audio_duration should use persisted provider usage metadata."""
        task = _transcribe_task(
            provider_meta={"usage": {"audio_duration_seconds": 30}},
        )

        assert task.audio_duration == 30

    def test_uses_chunk_end_time(self) -> None:
        """audio_duration should derive seconds from chunk metadata."""
        task = _transcribe_task(
            chunks=[
                ChunkMetadata(chunk_id=0, start_ms=0, end_ms=1000, file_path="a.wav"),
                ChunkMetadata(
                    chunk_id=1,
                    start_ms=1000,
                    end_ms=2500,
                    file_path="b.wav",
                ),
            ],
        )

        assert task.audio_duration == pytest.approx(2.5)

    def test_unknown_duration_is_zero(self) -> None:
        """audio_duration should not guess when no metadata is available."""
        task = _transcribe_task()

        assert task.audio_duration == pytest.approx(0.0)


@pytest.mark.unit
class TestTranscribeBase64Schema:
    """Tests for base64 transcription upload schema conversion."""

    def test_to_create_schema_builds_data_url(self) -> None:
        """Base64 payloads should be converted to data URLs."""
        data = TranscribeTaskBase64Schema(
            content_base64="ZmFrZQ==",
            mime_type="audio/wav",
            audio_duration_seconds=1.5,
        )

        create_schema = data.to_create_schema()

        assert create_schema.file_url == "data:audio/wav;base64,ZmFrZQ=="
        assert create_schema.audio_duration_seconds == pytest.approx(1.5)


@pytest.mark.unit
class TestCombineChunkTexts:
    """Tests for _combine_chunk_texts function."""

    def test_combines_texts_in_order(self) -> None:
        """_combine_chunk_texts should combine texts ordered by start_ms."""
        from pathlib import Path

        from apps.transcribe.chunker_ffmpeg import AudioChunk, ChunkTranscriptionResult

        chunk1 = AudioChunk(
            chunk_id=0, start_ms=0, end_ms=5000, file_path=Path("c0.wav")
        )
        chunk2 = AudioChunk(
            chunk_id=1, start_ms=5000, end_ms=10000, file_path=Path("c1.wav")
        )

        result1 = ChunkTranscriptionResult(
            chunk=chunk2, job_id="job2", text="Second part"
        )
        result2 = ChunkTranscriptionResult(
            chunk=chunk1, job_id="job1", text="First part"
        )

        combined = _combine_chunk_texts([result1, result2])

        assert "First part" in combined
        assert "Second part" in combined
        # First part should come before second part
        assert combined.index("First part") < combined.index("Second part")

    def test_skips_empty_texts(self) -> None:
        """_combine_chunk_texts should skip empty chunk texts."""
        from pathlib import Path

        from apps.transcribe.chunker_ffmpeg import AudioChunk, ChunkTranscriptionResult

        chunk1 = AudioChunk(
            chunk_id=0, start_ms=0, end_ms=5000, file_path=Path("c0.wav")
        )
        chunk2 = AudioChunk(
            chunk_id=1, start_ms=5000, end_ms=10000, file_path=Path("c1.wav")
        )

        result1 = ChunkTranscriptionResult(chunk=chunk1, job_id="job1", text="")
        result2 = ChunkTranscriptionResult(
            chunk=chunk2, job_id="job2", text="Real text"
        )

        combined = _combine_chunk_texts([result1, result2])

        assert combined == "Real text"

    def test_returns_empty_string_for_all_empty(self) -> None:
        """_combine_chunk_texts should return empty string when all texts are empty."""
        from pathlib import Path

        from apps.transcribe.chunker_ffmpeg import AudioChunk, ChunkTranscriptionResult

        chunk = AudioChunk(
            chunk_id=0, start_ms=0, end_ms=5000, file_path=Path("c0.wav")
        )
        result = ChunkTranscriptionResult(chunk=chunk, job_id="job1", text="")

        combined = _combine_chunk_texts([result])

        assert combined == ""

    def test_returns_empty_string_for_empty_list(self) -> None:
        """_combine_chunk_texts should return empty string for empty list."""
        combined = _combine_chunk_texts([])
        assert combined == ""


@pytest.mark.unit
class TestSaveError:
    """Tests for transcription save_error function."""

    async def test_sets_error_status(self) -> None:
        """save_error should set task status to error."""
        task = MagicMock()
        task.update_and_emit = AsyncMock()

        with patch(
            "apps.transcribe.services.conditions.Conditions",
        ) as mock_conditions_cls:
            mock_conditions = MagicMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            await save_error(task, "transcription failed")

        assert task.task_status == TaskStatusEnum.error
        task.update_and_emit.assert_awaited_once_with(
            task_report="transcription failed",
            log_type="error",
        )

    async def test_releases_condition(self) -> None:
        """save_error should release the condition for the task."""
        task = MagicMock()
        task.uid = "task_123"
        task.update_and_emit = AsyncMock()

        with patch(
            "apps.transcribe.services.conditions.Conditions",
        ) as mock_conditions_cls:
            mock_conditions = MagicMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            await save_error(task, "error")

        mock_conditions.release_condition.assert_called_once_with(task.uid)


@pytest.mark.unit
class TestSaveResult:
    """Tests for transcription save_result function."""

    async def test_sets_completed_status(self) -> None:
        """save_result should set task status to completed."""
        task = MagicMock()
        task.update_and_emit = AsyncMock()

        await save_result(task, "Transcribed text")

        assert task.task_status == TaskStatusEnum.completed

    async def test_normalizes_text(self) -> None:
        """save_result should normalize the result text."""
        task = MagicMock()
        task.update_and_emit = AsyncMock()

        await save_result(task, "  Text with spaces  ")

        assert task.result == "Text with spaces"

    async def test_saves_usage_info(self) -> None:
        """save_result should save usage amount and ID."""
        task = MagicMock()
        task.update_and_emit = AsyncMock()

        await save_result(task, "text", usage_amount=10.0, usage_id="usage_456")

        assert task.usage_amount == pytest.approx(10.0)
        assert task.usage_id == "usage_456"

    async def test_emits_webhook_after_result_is_stored(self) -> None:
        """save_result persists the result before the completion webhook emits."""
        task = MagicMock()
        task.update_and_emit = AsyncMock()

        await save_result(task, "Transcribed text")

        assert task.result == "Transcribed text"
        task.update_and_emit.assert_awaited_once_with(
            task_report="Task processed successfully",
        )


@pytest.mark.unit
class TestProcessTranscribe:
    """Tests for process_transcribe function."""

    async def test_returns_error_on_insufficient_quota(self) -> None:
        """process_transcribe should return error when quota is insufficient."""
        from apps.transcribe.services import process_transcribe

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.audio_duration = 10.0
        task.update_and_emit = AsyncMock()

        with (
            patch(
                "apps.transcribe.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=0,  # Insufficient quota
            ),
            patch(
                "apps.transcribe.services.conditions.Conditions",
            ) as mock_conditions_cls,
        ):
            mock_conditions = MagicMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            result = await process_transcribe(task)

        assert result.task_status == TaskStatusEnum.error

    async def test_falls_back_to_single_job_on_chunk_failure(self) -> None:
        """process_transcribe should fall back to single job when chunking fails."""
        from apps.transcribe.services import process_transcribe

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.audio_duration = 10.0
        task.update_and_emit = AsyncMock()
        task.save = AsyncMock(return_value=task)

        with (
            patch(
                "apps.transcribe.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=100,
            ),
            patch("apps.transcribe.services.Settings") as mock_settings,
            patch(
                "apps.transcribe.services._process_chunked_transcribe",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Chunking failed"),
            ),
            patch(
                "apps.transcribe.services._process_single_job",
                new_callable=AsyncMock,
                return_value=task,
            ) as mock_single,
        ):
            mock_settings.transcribe_enable_chunking = True
            mock_settings.transcribe_chunking_fallback_single = True

            await process_transcribe(task)

        mock_single.assert_called_once()

    async def test_does_not_fallback_when_disabled(self) -> None:
        """process_transcribe should not fall back when fallback is disabled."""
        from apps.transcribe.services import process_transcribe

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.audio_duration = 10.0
        task.update_and_emit = AsyncMock()

        with (
            patch(
                "apps.transcribe.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=100,
            ),
            patch("apps.transcribe.services.Settings") as mock_settings,
            patch(
                "apps.transcribe.services._process_chunked_transcribe",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Chunking failed"),
            ),
            patch(
                "apps.transcribe.services.conditions.Conditions",
            ) as mock_conditions_cls,
        ):
            mock_settings.transcribe_enable_chunking = True
            mock_settings.transcribe_chunking_fallback_single = False
            mock_conditions = MagicMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            result = await process_transcribe(task)

        assert result.task_status == TaskStatusEnum.error


@pytest.mark.unit
class TestTranscriptionQuotaAndMetering:
    """Tests for transcription quota checking and usage metering."""

    async def test_checks_quota_before_processing(self) -> None:
        """process_transcribe should check quota before starting transcription."""
        from apps.transcribe.services import process_transcribe

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.audio_duration = 10.0
        task.update_and_emit = AsyncMock()

        with (
            patch(
                "apps.transcribe.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=100,
            ) as mock_check_quota,
            patch("apps.transcribe.services.Settings") as mock_settings,
            patch(
                "apps.transcribe.services._process_single_job",
                new_callable=AsyncMock,
                return_value=task,
            ),
        ):
            mock_settings.transcribe_enable_chunking = False

            await process_transcribe(task)

        mock_check_quota.assert_called_once_with(
            task.user_id,
            task.audio_duration / 60,
            raise_exception=False,
        )

    async def test_meters_usage_after_chunked_transcription(self) -> None:
        """Chunked transcription should meter usage after completion."""
        from pathlib import Path

        from apps.transcribe.chunker_ffmpeg import (
            AudioChunk,
            ChunkPlan,
            ChunkTranscriptionResult,
        )
        from apps.transcribe.services import _process_chunked_transcribe

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.file_url = "https://example.com/audio.mp3"
        task.update_and_emit = AsyncMock()
        task.save = AsyncMock(return_value=task)

        # Create mock chunks
        chunk1 = AudioChunk(
            chunk_id=0, start_ms=0, end_ms=5000, file_path=Path("c0.wav")
        )
        chunk2 = AudioChunk(
            chunk_id=1, start_ms=5000, end_ms=10000, file_path=Path("c1.wav")
        )

        chunk_plan = ChunkPlan(
            duration_ms=10000, chunks=[chunk1, chunk2], workspace=Path("test")
        )

        # Mock chunk results with costs
        result1 = ChunkTranscriptionResult(
            chunk=chunk1,
            job_id="job1",
            text="First part",
            audio_duration_ms=5000,
            transcription_cost=5.0,
        )
        result2 = ChunkTranscriptionResult(
            chunk=chunk2,
            job_id="job2",
            text="Second part",
            audio_duration_ms=5000,
            transcription_cost=5.0,
        )

        with (
            patch(
                "apps.transcribe.services.chunker.create_chunk_plan",
                new_callable=AsyncMock,
                return_value=chunk_plan,
            ),
            patch(
                "apps.transcribe.services._transcribe_chunks",
                new_callable=AsyncMock,
                return_value=[result1, result2],
            ),
            patch(
                "apps.transcribe.services.finance.meter_cost",
                new_callable=AsyncMock,
            ) as mock_meter,
            patch(
                "apps.transcribe.services.conditions.Conditions",
            ) as mock_conditions_cls,
        ):
            mock_conditions = MagicMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            await _process_chunked_transcribe(task, sync=False)

        # Verify metering was called with total cost
        mock_meter.assert_called_once_with(task.user_id, 10.0)

    async def test_meters_usage_after_webhook_processing(self) -> None:
        """Webhook processing should meter usage after transcription completion."""
        from soniox.types import TranscriptionJobStatus

        from apps.transcribe.services import process_transcription_webhook

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.transcription_job_id = "job_123"
        task.task_status = TaskStatusEnum.processing
        task.update_and_emit = AsyncMock()
        task.save = AsyncMock(return_value=task)

        webhook_data = MagicMock()
        webhook_data.id = "job_123"
        webhook_data.status = TranscriptionJobStatus.COMPLETED

        mock_job_result = MagicMock()
        mock_job_result.audio_duration_ms = 60000  # 1 minute

        mock_transcript = MagicMock()
        mock_transcript.text = "Transcribed text"

        with (
            patch("apps.transcribe.services.Settings") as mock_settings,
            patch(
                "apps.transcribe.services.soniox.get_transcription_job_async",
                new_callable=AsyncMock,
                return_value=mock_job_result,
            ),
            patch(
                "apps.transcribe.services.soniox.get_transcription_result_async",
                new_callable=AsyncMock,
                return_value=mock_transcript,
            ),
            patch(
                "apps.transcribe.services.finance.meter_cost",
                new_callable=AsyncMock,
            ) as mock_meter,
            patch(
                "apps.transcribe.services.conditions.Conditions",
            ) as mock_conditions_cls,
        ):
            mock_settings.minutes_price = 10
            mock_conditions = MagicMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            await process_transcription_webhook(task, webhook_data)

        # Verify metering was called
        mock_meter.assert_called_once()
        # Cost should be ceil((60000 / 60 / 1000) * 10) = 10
        assert mock_meter.call_args[0][1] == pytest.approx(1.0)


@pytest.mark.unit
class TestTranscriptionErrorHandling:
    """Tests for transcription error handling."""

    async def test_handles_transcription_job_failure(self) -> None:
        """process_transcribe should handle transcription job failures gracefully."""
        from soniox.types import TranscriptionJobStatus

        from apps.transcribe.services import _process_single_job

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.tenant_id = "tenant_123"
        task.file_url = "https://example.com/audio.mp3"
        task.transcription_job_id = "job_123"
        task.save = AsyncMock(return_value=task)
        task.update_and_emit = AsyncMock()

        mock_job = MagicMock()
        mock_job.id = "job_123"
        mock_job.status = TranscriptionJobStatus.ERROR
        mock_job.error_message = "Transcription failed"

        with (
            patch(
                "apps.transcribe.services.soniox.transcribe_url_async",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch(
                "apps.transcribe.services.soniox.get_transcription_job_async",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch(
                "apps.transcribe.services.conditions.Conditions",
            ) as mock_conditions_cls,
        ):
            mock_conditions = MagicMock()
            mock_conditions.wait_condition = AsyncMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            result = await _process_single_job(task, sync=True)

        assert result.task_status == TaskStatusEnum.error

    async def test_handles_chunk_transcription_failure(self) -> None:
        """Chunked transcription should handle individual chunk failures."""
        from pathlib import Path

        from soniox.types import TranscriptionJobStatus

        from apps.transcribe.chunker_ffmpeg import AudioChunk, ChunkPlan
        from apps.transcribe.services import _transcribe_chunks

        task = MagicMock()
        task.uid = "task_123"

        chunk = AudioChunk(
            chunk_id=0, start_ms=0, end_ms=5000, file_path=Path("c0.wav")
        )
        chunk_plan = ChunkPlan(duration_ms=5000, chunks=[chunk], workspace=Path("test"))

        mock_job = MagicMock()
        mock_job.id = "job_123"
        mock_job.status = TranscriptionJobStatus.ERROR
        mock_job.error_message = "Chunk transcription failed"

        with (
            patch("apps.transcribe.services.Settings") as mock_settings,
            patch(
                "apps.transcribe.services.soniox.transcribe_file_async",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch(
                "apps.transcribe.services._wait_for_job_completion",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
        ):
            mock_settings.transcribe_max_parallel_requests = 1

            with pytest.raises(RuntimeError, match="Chunk 0 failed"):
                await _transcribe_chunks(task, chunk_plan)

    async def test_saves_error_message_on_failure(self) -> None:
        """Transcription errors should save descriptive error messages."""
        from apps.transcribe.services import save_error

        task = MagicMock()
        task.uid = "task_123"
        task.update_and_emit = AsyncMock()

        with patch(
            "apps.transcribe.services.conditions.Conditions",
        ) as mock_conditions_cls:
            mock_conditions = MagicMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            await save_error(task, "transcription_job_failed")

        task.update_and_emit.assert_awaited_once_with(
            task_report="transcription_job_failed",
            log_type="error",
        )
        assert task.task_status == TaskStatusEnum.error

    async def test_handles_webhook_error_status(self) -> None:
        """Webhook processing should handle error status from transcription service."""
        from soniox.types import TranscriptionJobStatus

        from apps.transcribe.services import process_transcription_webhook

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.transcription_job_id = "job_123"
        task.update_and_emit = AsyncMock()

        webhook_data = MagicMock()
        webhook_data.id = "job_123"
        webhook_data.status = TranscriptionJobStatus.ERROR

        mock_job = MagicMock()
        mock_job.error_message = "Transcription service error"

        with (
            patch(
                "apps.transcribe.services.soniox.get_transcription_job_async",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch(
                "apps.transcribe.services.conditions.Conditions",
            ) as mock_conditions_cls,
        ):
            mock_conditions = MagicMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            result = await process_transcription_webhook(task, webhook_data)

        assert result.task_status == TaskStatusEnum.error

    async def test_handles_mismatched_job_id_in_webhook(self) -> None:
        """Webhook processing should reject webhooks with mismatched job IDs."""
        from soniox.types import TranscriptionJobStatus

        from apps.transcribe.services import process_transcription_webhook

        task = MagicMock()
        task.uid = "task_123"
        task.user_id = "user_123"
        task.transcription_job_id = "job_123"
        task.update_and_emit = AsyncMock()

        webhook_data = MagicMock()
        webhook_data.id = "different_job_456"
        webhook_data.status = TranscriptionJobStatus.COMPLETED

        mock_job = MagicMock()
        mock_job.error_message = None

        with (
            patch(
                "apps.transcribe.services.soniox.get_transcription_job_async",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch(
                "apps.transcribe.services.conditions.Conditions",
            ) as mock_conditions_cls,
        ):
            mock_conditions = MagicMock()
            mock_conditions.release_condition = AsyncMock()
            mock_conditions_cls.return_value = mock_conditions

            result = await process_transcription_webhook(task, webhook_data)

        assert result.task_status == TaskStatusEnum.error
        task.update_and_emit.assert_awaited_once()


@pytest.mark.unit
class TestTranscribeFileContent:
    """Tests for TranscribeTaskSchemaCreate.file_content."""

    async def test_returns_cached_content(self) -> None:
        """file_content should return cached _file_content if set."""
        from io import BytesIO

        from apps.transcribe.schemas import TranscribeTaskSchemaCreate

        schema = TranscribeTaskSchemaCreate(file_url="https://example.com/a.mp3")
        cached = BytesIO(b"cached data")
        schema._file_content = cached

        result = await schema.file_content()
        assert result is cached

    async def test_decodes_base64_data_url(self, mock_audio_bytes: bytes) -> None:
        """file_content should decode base64-encoded data URLs."""
        import base64

        from apps.transcribe.schemas import TranscribeTaskSchemaCreate

        encoded = base64.b64encode(mock_audio_bytes).decode("utf-8")
        schema = TranscribeTaskSchemaCreate(file_url=f"data:audio/wav;base64,{encoded}")

        content = await schema.file_content()
        assert content.read() == mock_audio_bytes

    async def test_handles_invalid_base64(self) -> None:
        """file_content should not crash on invalid base64 data URLs."""
        from apps.transcribe.schemas import TranscribeTaskSchemaCreate

        schema = TranscribeTaskSchemaCreate(
            file_url="data:audio/wav;base64,!!!invalid!!!"
        )

        content = await schema.file_content()
        assert content.read() == b""

    async def test_fetches_from_http_url(self) -> None:
        """file_content should fetch content from HTTP URLs."""
        from unittest.mock import AsyncMock, patch

        from apps.transcribe.schemas import TranscribeTaskSchemaCreate

        schema = TranscribeTaskSchemaCreate(file_url="https://example.com/a.mp3")

        with patch(
            "apps.transcribe.schemas.download_bytes",
            new_callable=AsyncMock,
            return_value=__import__("io").BytesIO(b"audio data"),
        ):
            content = await schema.file_content()
            assert content.read() == b"audio data"


@pytest.mark.unit
class TestTranscribeFileContentBase64:
    """Tests for TranscribeTaskSchemaCreate.file_content_base64."""

    async def test_returns_base64_encoded_content(
        self, mock_audio_bytes: bytes
    ) -> None:
        """file_content_base64 should return base64 string of content."""
        import base64

        from apps.transcribe.schemas import TranscribeTaskSchemaCreate

        encoded = base64.b64encode(mock_audio_bytes).decode("utf-8")
        schema = TranscribeTaskSchemaCreate(file_url=f"data:audio/wav;base64,{encoded}")

        result = await schema.file_content_base64()
        assert result == encoded


@pytest.mark.unit
class TestTranscribeUploadFormSchema:
    """Tests for TranscribeTaskUploadFormSchema.as_form."""

    def test_as_form_parses_fields(self) -> None:
        """as_form should parse form fields correctly."""
        from apps.transcribe.schemas import TranscribeTaskUploadFormSchema

        result = TranscribeTaskUploadFormSchema.as_form(
            audio_duration_seconds=30.0,
            provider="soniox",
            model="whisper-1",
            user_id="user_123",
            webhook_url="https://hook.example.com",
        )

        assert result.audio_duration_seconds == pytest.approx(30.0)
        assert result.provider == "soniox"
        assert result.model == "whisper-1"
        assert result.user_id == "user_123"
        assert result.webhook_url == "https://hook.example.com"

    def test_as_form_uses_defaults(self) -> None:
        """as_form should use defaults for missing fields."""
        from apps.transcribe.schemas import TranscribeTaskUploadFormSchema

        result = TranscribeTaskUploadFormSchema.as_form(
            audio_duration_seconds=None,
            provider="soniox",
            model=None,
            user_id=None,
            webhook_url=None,
        )

        assert result.provider == "soniox"
        assert result.audio_duration_seconds is None


@pytest.mark.unit
class TestTranscribeBase64ToCreateSchema:
    """Tests for TranscribeTaskBase64Schema.to_create_schema."""

    def test_to_create_schema_builds_data_url(self) -> None:
        """Should build a data URL from base64 content."""
        from apps.transcribe.schemas import TranscribeTaskBase64Schema

        schema = TranscribeTaskBase64Schema(
            content_base64="ZmFrZQ==",
            mime_type="audio/wav",
            audio_duration_seconds=2.0,
            provider="soniox",
        )

        create = schema.to_create_schema()
        assert create.file_url == "data:audio/wav;base64,ZmFrZQ=="
        assert create.audio_duration_seconds == pytest.approx(2.0)
        assert create.provider == "soniox"

    def test_to_create_schema_preserves_data_url(self) -> None:
        """Should not re-wrap if already a data URL."""
        from apps.transcribe.schemas import TranscribeTaskBase64Schema

        schema = TranscribeTaskBase64Schema(
            content_base64="data:audio/wav;base64,ZmFrZQ==",
        )

        create = schema.to_create_schema()
        assert create.file_url == "data:audio/wav;base64,ZmFrZQ=="


@pytest.mark.unit
class TestTranscribeAudioDurationMetaData:
    """Tests for TranscribeTaskSchema audio_duration with meta_data fallback."""

    def test_uses_meta_data_audio_duration_seconds(self) -> None:
        """audio_duration should fall back to meta_data.audio_duration_seconds."""

        task = _transcribe_task(meta_data={"audio_duration_seconds": 42.5})

        assert task.audio_duration == pytest.approx(42.5)

    def test_uses_meta_data_audio_duration_ms(self) -> None:
        """audio_duration should fall back to meta_data.audio_duration_ms."""

        task = _transcribe_task(meta_data={"audio_duration_ms": 55000})

        assert task.audio_duration == pytest.approx(55.0)
