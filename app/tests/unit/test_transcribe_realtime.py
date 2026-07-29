"""Unit tests for realtime Soniox WebSocket proxy."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from usso.exceptions import USSOException

from apps.transcribe import realtime, ws_auth
from apps.transcribe.realtime import (
    AudioBuffer,
    audio_filename_for_format,
    build_soniox_config,
    duration_seconds_from_proc_ms,
    meter_realtime_session,
    normalize_soniox_message,
    persist_realtime_session,
)


@pytest.mark.unit
class TestWsAuth:
    """Tests for WebSocket USSO authentication helpers."""

    def test_query_token_injected_into_headers(self) -> None:
        """Browser query access_token should become Authorization Bearer."""
        websocket = MagicMock()
        websocket.query_params = {"access_token": "tok123"}
        websocket.headers = {"host": "example.com"}
        websocket.cookies = {}

        view = ws_auth.websocket_auth_view(websocket)

        assert view.headers.get("Authorization") == "Bearer tok123"

    def test_query_api_key_injected(self) -> None:
        """Query api_key should become x-api-key when header missing."""
        websocket = MagicMock()
        websocket.query_params = {"api_key": "key-1"}
        websocket.headers = {}
        websocket.cookies = {}

        view = ws_auth.websocket_auth_view(websocket)

        assert view.headers.get("x-api-key") == "key-1"

    def test_no_query_returns_websocket(self) -> None:
        """Without query credentials the raw websocket is used."""
        websocket = MagicMock()
        websocket.query_params = {}

        assert ws_auth.websocket_auth_view(websocket) is websocket

    def test_authenticate_websocket_calls_usso(self) -> None:
        """authenticate_websocket should delegate to jwt_access_security_ws."""
        websocket = MagicMock()
        websocket.query_params = {}
        user = SimpleNamespace(uid="u1", tenant_id="t1")
        usso = MagicMock()
        usso.jwt_access_security_ws.return_value = user

        with patch("apps.transcribe.ws_auth.get_usso", return_value=usso):
            result = ws_auth.authenticate_websocket(websocket)

        assert result is user
        usso.jwt_access_security_ws.assert_called_once()

    def test_authenticate_websocket_none_raises(self) -> None:
        """Missing user from USSO should raise unauthorized."""
        websocket = MagicMock()
        websocket.query_params = {}
        usso = MagicMock()
        usso.jwt_access_security_ws.return_value = None

        with (
            patch("apps.transcribe.ws_auth.get_usso", return_value=usso),
            pytest.raises(USSOException),
        ):
            ws_auth.authenticate_websocket(websocket)


@pytest.mark.unit
class TestRealtimeHelpers:
    """Pure helper coverage for realtime bridge."""

    def test_build_soniox_config_injects_api_key_and_strips_client_key(self) -> None:
        """Server api_key must win; client api_key must not pass through."""
        with (
            patch.object(realtime.Settings, "soniox_rt_model", "stt-rt-v5"),
            patch.object(realtime.Settings, "soniox_rt_language_hints", ["fa", "en"]),
        ):
            cfg = build_soniox_config(
                {"api_key": "leaked", "audio_format": "mp3", "model": "stt-rt-v5"},
                api_key="server-key",
                client_reference_id="user-1",
            )

        assert cfg["api_key"] == "server-key"
        assert cfg["client_reference_id"] == "user-1"
        assert cfg["audio_format"] == "mp3"
        assert "leaked" not in cfg.values()

    def test_normalize_partial_and_final(self) -> None:
        """Tokens should split into partial and final client messages."""
        messages = normalize_soniox_message({
            "tokens": [
                {"text": "Hi", "is_final": True},
                {"text": " there", "is_final": False},
            ],
            "final_audio_proc_ms": 100,
            "total_audio_proc_ms": 200,
        })

        assert messages[0] == {
            "type": "final",
            "text": "Hi",
            "tokens": [{"text": "Hi", "is_final": True}],
        }
        assert messages[1]["type"] == "partial"
        assert messages[1]["text"] == " there"

    def test_normalize_finished_and_error(self) -> None:
        """Finished and error payloads should map to client types."""
        finished = normalize_soniox_message({
            "finished": True,
            "final_audio_proc_ms": 1500,
            "total_audio_proc_ms": 1600,
        })
        assert finished == [
            {
                "type": "finished",
                "final_audio_proc_ms": 1500,
                "total_audio_proc_ms": 1600,
            },
        ]

        errors = normalize_soniox_message({
            "error_code": 400,
            "error_message": "bad",
        })
        assert errors[0]["type"] == "error"
        assert "400" in errors[0]["detail"]

    def test_audio_filename_and_duration(self) -> None:
        """Filename and metering duration helpers."""
        assert audio_filename_for_format("webm") == "realtime.webm"
        assert audio_filename_for_format("pcm_s16le") == "realtime.pcm"
        assert duration_seconds_from_proc_ms(500, None) == pytest.approx(1.0)
        assert duration_seconds_from_proc_ms(2500, 3000) == pytest.approx(2.5)

    def test_audio_buffer_limit(self) -> None:
        """Buffer must reject frames that exceed the configured ceiling."""
        buf = AudioBuffer(4)
        buf.append(b"ab")
        with pytest.raises(realtime.AudioBufferError):
            buf.append(b"cdef")
        assert buf.getvalue() == b"ab"


@pytest.mark.unit
class TestPersistAndMeter:
    """Persistence and billing for realtime sessions."""

    @pytest.mark.asyncio
    async def test_persist_uploads_and_creates_task(self) -> None:
        """Successful persist should upload audio and create TranscribeTask."""
        user = SimpleNamespace(uid="u1", tenant_id="t1")
        task = SimpleNamespace(uid="task-1")

        with (
            patch(
                "apps.transcribe.realtime.media.upload_file",
                new_callable=AsyncMock,
                return_value="https://media.example/a.webm",
            ) as upload,
            patch(
                "apps.transcribe.realtime.TranscribeTask.create_item",
                new_callable=AsyncMock,
                return_value=task,
            ) as create,
        ):
            result, url, err = await persist_realtime_session(
                user=user,
                audio=b"audio-bytes",
                audio_format="webm",
                model="stt-rt-v5",
                transcript="hello",
                final_audio_proc_ms=2000,
                total_audio_proc_ms=2100,
            )

        assert result is task
        assert url == "https://media.example/a.webm"
        assert err is None
        upload.assert_awaited_once()
        create.assert_awaited_once()
        kwargs = create.await_args.args[0]
        assert kwargs["meta_data"]["realtime"] is True
        assert kwargs["result"] == "hello"
        assert kwargs["file_url"] == url

    @pytest.mark.asyncio
    async def test_persist_upload_failure(self) -> None:
        """Upload failure should return an error without creating a task."""
        user = SimpleNamespace(uid="u1", tenant_id="t1")

        with (
            patch(
                "apps.transcribe.realtime.media.upload_file",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "apps.transcribe.realtime.TranscribeTask.create_item",
                new_callable=AsyncMock,
            ) as create,
        ):
            result, url, err = await persist_realtime_session(
                user=user,
                audio=b"x",
                audio_format="auto",
                model=None,
                transcript="",
                final_audio_proc_ms=None,
                total_audio_proc_ms=None,
            )

        assert result is None
        assert url is None
        assert err is not None
        create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_meter_realtime_session(self) -> None:
        """Metering should estimate cost from processed audio ms."""
        with (
            patch(
                "apps.transcribe.realtime.finance.estimate_transcribe_cost",
                return_value=1.5,
            ) as estimate,
            patch(
                "apps.transcribe.realtime.finance.meter_cost",
                new_callable=AsyncMock,
            ) as meter,
        ):
            await meter_realtime_session(
                user_id="u1",
                model="stt-rt-v5",
                final_audio_proc_ms=60000,
                total_audio_proc_ms=61000,
                task_uid="task-1",
            )

        estimate.assert_called_once()
        meter.assert_awaited_once()
        assert meter.await_args.args[0] == "u1"
        assert meter.await_args.args[1] == pytest.approx(1.5)


class _FakeSonioxWs:
    """Minimal async context manager mimicking Soniox WS."""

    def __init__(self, messages: list[str]) -> None:
        self.sent: list[object] = []
        self._messages = messages

    async def __aenter__(self) -> _FakeSonioxWs:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def send(self, data: object) -> None:
        self.sent.append(data)

    def __aiter__(self) -> _FakeSonioxWs:
        self._iter = iter(self._messages)
        return self

    async def __anext__(self) -> str:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _queued_receive(*messages: dict) -> AsyncMock:
    """Return an AsyncMock that yields messages then hangs until cancelled."""
    queue = list(messages)
    hang = asyncio.Event()

    async def _receive() -> dict:
        if queue:
            return queue.pop(0)
        await hang.wait()
        return {"type": "websocket.disconnect", "code": 1000}

    return AsyncMock(side_effect=_receive)


@pytest.mark.unit
class TestHandleRealtimeSession:
    """Orchestration tests for the WebSocket handler."""

    @pytest.mark.asyncio
    async def test_rejects_unauthorized(self) -> None:
        """Auth failure should close with 4401 and never accept."""
        websocket = MagicMock()
        websocket.close = AsyncMock()
        websocket.accept = AsyncMock()

        with patch(
            "apps.transcribe.realtime.authenticate_websocket",
            side_effect=USSOException(401, "unauthorized"),
        ):
            await realtime.handle_realtime_session(websocket)

        websocket.close.assert_awaited_once_with(code=4401)
        websocket.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_missing_soniox_key(self) -> None:
        """Missing SONIOX_API_KEY should accept then close 4403."""
        websocket = MagicMock()
        websocket.close = AsyncMock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        user = SimpleNamespace(uid="u1", tenant_id="t1")

        with (
            patch(
                "apps.transcribe.realtime.authenticate_websocket",
                return_value=user,
            ),
            patch.object(realtime.Settings, "soniox_api_key", None),
        ):
            await realtime.handle_realtime_session(websocket)

        websocket.accept.assert_awaited_once()
        websocket.close.assert_awaited_once_with(code=4403)
        sent = websocket.send_text.await_args.args[0]
        assert "SONIOX_API_KEY" in sent

    @pytest.mark.asyncio
    async def test_bridge_end_to_end_with_mocks(self) -> None:
        """Config + audio + Soniox finished should persist and meter."""
        websocket = MagicMock()
        websocket.accept = AsyncMock()
        websocket.close = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock(
            return_value='{"audio_format":"webm","model":"stt-rt-v5"}',
        )
        websocket.receive = _queued_receive(
            {"type": "websocket.receive", "bytes": b"\x01\x02"},
            {"type": "websocket.receive", "text": '{"type":"end"}'},
        )

        fake_soniox = _FakeSonioxWs([
            (
                '{"tokens":[{"text":"hello","is_final":true}],'
                '"final_audio_proc_ms":1000,"total_audio_proc_ms":1100}'
            ),
            '{"finished":true,"final_audio_proc_ms":1200,"total_audio_proc_ms":1300}',
        ])
        user = SimpleNamespace(uid="user-1", tenant_id="tenant-1")
        task = SimpleNamespace(uid="task-99")

        with (
            patch(
                "apps.transcribe.realtime.authenticate_websocket",
                return_value=user,
            ),
            patch.object(realtime.Settings, "soniox_api_key", "sk-test"),
            patch.object(realtime.Settings, "soniox_ws_url", "wss://example/ws"),
            patch.object(realtime.Settings, "soniox_rt_model", "stt-rt-v5"),
            patch.object(realtime.Settings, "soniox_rt_language_hints", ["fa", "en"]),
            patch.object(realtime.Settings, "soniox_rt_max_buffer_bytes", 1024),
            patch(
                "apps.transcribe.realtime.websockets.connect",
                return_value=fake_soniox,
            ),
            patch(
                "apps.transcribe.realtime.media.upload_file",
                new_callable=AsyncMock,
                return_value="https://media/x.webm",
            ),
            patch(
                "apps.transcribe.realtime.TranscribeTask.create_item",
                new_callable=AsyncMock,
                return_value=task,
            ),
            patch(
                "apps.transcribe.realtime.finance.estimate_transcribe_cost",
                return_value=0.1,
            ),
            patch(
                "apps.transcribe.realtime.finance.meter_cost",
                new_callable=AsyncMock,
            ) as meter,
        ):
            await realtime.handle_realtime_session(websocket)

        assert any(
            isinstance(item, str) and "sk-test" in item for item in fake_soniox.sent
        )
        assert b"\x01\x02" in fake_soniox.sent
        meter.assert_awaited_once()

        payloads = [call.args[0] for call in websocket.send_text.await_args_list]
        assert any('"type": "final"' in p or '"type":"final"' in p for p in payloads)
        assert any("task-99" in p for p in payloads)
        assert any("finished" in p for p in payloads)
