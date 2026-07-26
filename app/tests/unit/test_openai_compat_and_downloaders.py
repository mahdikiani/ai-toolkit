# Unit tests for OpenAI-compat audio routes and downloader/archive helpers.

from __future__ import annotations

import gzip
import zipfile
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.responses import Response

from apps.openai_compat import audio as openai_audio
from apps.openai_compat import services as openai_services
from apps.youtube import services as youtube_services
from utils.downloaders import web as web_downloader
from utils.files import archive_utils


@pytest.mark.unit
class TestOpenAICompatServices:
    def test_openai_error_shape(self) -> None:
        err = openai_services.openai_error(400, "bad", "msg")
        assert err.status_code == 400
        assert err.error_code == "bad"

    def test_estimate_chat_cost_uses_model_pricing(self) -> None:
        body = {"model": "openai/gpt-4o-mini", "messages": [{"content": "hello"}]}
        cost = openai_services.estimate_chat_cost(body)
        assert cost > 0

    async def test_handle_non_stream_chat_success(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "model": "m",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
        with (
            patch(
                "apps.openai_compat.services.post_chat_completion_unchecked",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch(
                "apps.openai_compat.services.finance.check_quota",
                new_callable=AsyncMock,
            ),
            patch(
                "apps.openai_compat.services.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            result = await openai_services.handle_non_stream_chat(
                {"model": "m", "messages": [{"role": "user", "content": "x"}]},
                user_id="u1",
                model="m",
            )
        assert result.status_code == 200
        assert b"chatcmpl-" in result.body

    async def test_list_models_requires_auth(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/openai/v1/models")
        assert response.status_code in (401, 403)


@pytest.mark.unit
class TestOpenAICompatAudio:
    async def test_create_speech_rejects_missing_input(self) -> None:
        with pytest.raises(Exception) as exc:
            await openai_audio.create_speech({}, user_id="u1")
        assert exc.value.status_code == 400

    async def test_create_speech_proxies_openrouter(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"audio-bytes"

        with (
            patch(
                "apps.openai_compat.audio.finance.check_quota",
                new_callable=AsyncMock,
            ),
            patch("apps.openai_compat.audio.resolve_api_key"),
            patch("apps.openai_compat.audio.build_headers", return_value={}),
            patch(
                "apps.openai_compat.audio.httpx.AsyncClient",
            ) as client_cls,
            patch(
                "apps.openai_compat.audio.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.__aexit__.return_value = None
            client.post = AsyncMock(return_value=mock_resp)
            client_cls.return_value = client

            result = await openai_audio.create_speech(
                {"input": "hello", "response_format": "mp3"},
                user_id="u1",
            )

        assert isinstance(result, Response)
        assert result.body == b"audio-bytes"
        assert result.media_type == "audio/mpeg"

    async def test_create_transcription_requires_content(self) -> None:
        with pytest.raises(Exception) as exc:
            await openai_audio.create_transcription(
                b"", filename="a.wav", user_id="u1"
            )
        assert exc.value.status_code == 400

    async def test_create_transcription_without_soniox_key(self) -> None:
        with (
            patch.object(openai_audio.Settings, "soniox_api_key", None),
            patch(
                "apps.openai_compat.audio.finance.check_quota",
                new_callable=AsyncMock,
            ),
            pytest.raises(Exception) as exc,
        ):
            await openai_audio.create_transcription(
                b"abc", filename="a.wav", user_id="u1"
            )
        assert exc.value.status_code == 503


@pytest.mark.unit
class TestOpenAICompatRoutes:
    async def test_audio_speech_route_invalid_json(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/openai/v1/audio/speech",
            content=b"{bad",
            headers={"Content-Type": "application/json", "x-api-key": "test"},
        )
        assert response.status_code in (400, 401, 403, 422)

    async def test_audio_transcriptions_route_without_auth(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/openai/v1/audio/transcriptions",
            files={"file": ("a.wav", b"abc", "audio/wav")},
        )
        assert response.status_code in (401, 403)


@pytest.mark.unit
class TestWebDownloader:
    async def test_download_bytes_follows_redirect(self) -> None:
        first_req = httpx.Request("GET", "https://example.com/start")
        second_req = httpx.Request("GET", "https://example.com/final")
        first = httpx.Response(
            302,
            headers={"location": "https://example.com/final"},
            request=first_req,
        )
        second = httpx.Response(200, content=b"payload", request=second_req)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[first, second])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("utils.downloaders.web.assert_safe_url"),
            patch("utils.downloaders.web.is_gdrive_url", return_value=False),
            patch(
                "utils.downloaders.web.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            buf = await web_downloader.download_bytes("https://example.com/start")

        assert buf.read() == b"payload"


@pytest.mark.unit
class TestArchiveUtilsExtended:
    def test_extract_zip_roundtrip(self, tmp_path) -> None:
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("nested/a.txt", b"hello")
        buf.seek(0)
        result = archive_utils.extract_archive(buf, "application/zip")
        assert result is not None
        temp_dir, paths = result
        assert any(p.name == "a.txt" for p in paths)
        assert (temp_dir / "nested" / "a.txt").read_bytes() == b"hello"

    def test_extract_gzip(self) -> None:
        payload = gzip.compress(b"plain")
        buf = BytesIO(payload)
        result = archive_utils.extract_archive(buf, "application/gzip")
        assert result is not None
        _, paths = result
        assert paths[0].read_bytes() == b"plain"

    def test_compress_directory_to_zip(self, tmp_path) -> None:
        sample = tmp_path / "a.txt"
        sample.write_text("x", encoding="utf-8")
        out = archive_utils.compress_directory_to_zip(tmp_path)
        with zipfile.ZipFile(out) as zf:
            assert "a.txt" in zf.namelist()


@pytest.mark.unit
class TestYoutubeServices:
    async def test_process_youtube_missing_api_key(self) -> None:
        task = MagicMock()
        task.video_id = "abc123"
        task.user_id = "u1"
        task.update_and_emit = AsyncMock(return_value=task)

        with patch.object(youtube_services.Settings, "youtube_transcript_api_key", None):
            result = await youtube_services.process_youtube(task)

        task.update_and_emit.assert_awaited_once()
        assert result is task

    async def test_process_youtube_success(self) -> None:
        task = MagicMock()
        task.video_id = "https://youtu.be/abc123"
        task.user_id = "u1"
        task.update_and_emit = AsyncMock(return_value=task)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"tracks": [{"transcript": [{"text": "hello"}, {"text": "world"}]}]}
        ]
        mock_response.raise_for_status = MagicMock()

        with (
            patch.object(
                youtube_services.Settings, "youtube_transcript_api_key", "key"
            ),
            patch(
                "apps.youtube.services.httpx.AsyncClient",
            ) as client_cls,
            patch(
                "apps.youtube.services.finance.estimate_youtube_cost",
                return_value=1.0,
            ),
            patch(
                "apps.youtube.services.finance.meter_cost",
                new_callable=AsyncMock,
                return_value=MagicMock(amount=1.0, uid="usage-1"),
            ),
        ):
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.__aexit__.return_value = None
            client.post = AsyncMock(return_value=mock_response)
            client_cls.return_value = client

            result = await youtube_services.process_youtube(task)

        assert result is task
        task.update_and_emit.assert_awaited()
        assert task.update_and_emit.await_args.kwargs["task_status"].value == "completed"
