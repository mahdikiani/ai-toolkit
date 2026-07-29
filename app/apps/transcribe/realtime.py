"""Real-time Soniox WebSocket proxy with USSO auth, persist, and metering."""

from __future__ import annotations

import asyncio
import json
import logging
from io import BytesIO
from typing import Any

import websockets
from fastapi import WebSocket, WebSocketDisconnect
from fastapi_mongo_base.tasks import TaskStatusEnum
from usso import UserData
from usso.exceptions import USSOException
from websockets.asyncio.client import ClientConnection

from server.config import Settings
from utils.billing import finance
from utils.integrations import media

from .models import TranscribeTask
from .ws_auth import authenticate_websocket

logger = logging.getLogger(__name__)

SONIOX_WS_CLOSE_UNAUTHORIZED = 4401
SONIOX_WS_CLOSE_FORBIDDEN = 4403

_CONFIG_KEYS = frozenset({
    "audio_format",
    "model",
    "language_hints",
    "enable_endpoint_detection",
    "enable_speaker_diarization",
    "enable_language_identification",
    "sample_rate",
    "num_channels",
    "context",
    "translation",
    "max_endpoint_delay_ms",
    "endpoint_sensitivity",
    "endpoint_latency_adjustment_level",
})

_AUDIO_EXTENSIONS = {
    "auto": ".webm",
    "webm": ".webm",
    "mp3": ".mp3",
    "wav": ".wav",
    "ogg": ".ogg",
    "flac": ".flac",
    "aac": ".aac",
}


class AudioBufferError(BufferError):
    """Raised when the live audio buffer exceeds its configured limit."""


def build_soniox_config(
    client_config: dict[str, Any],
    *,
    api_key: str,
    client_reference_id: str,
) -> dict[str, Any]:
    """Build upstream Soniox config; inject api_key server-side only."""
    cfg = {key: client_config[key] for key in _CONFIG_KEYS if key in client_config}
    cfg["api_key"] = api_key
    cfg.setdefault("model", Settings.soniox_rt_model)
    cfg.setdefault("audio_format", "auto")
    cfg.setdefault(
        "language_hints",
        list(Settings.soniox_rt_language_hints),
    )
    cfg.setdefault("enable_endpoint_detection", True)
    cfg.setdefault("enable_language_identification", True)
    cfg.setdefault("enable_speaker_diarization", False)
    cfg["client_reference_id"] = client_reference_id
    return cfg


def normalize_soniox_message(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Map one Soniox JSON message to zero or more client-facing messages."""
    if payload.get("error_code") is not None:
        detail = (
            f"{payload.get('error_code')}: "
            f"{payload.get('error_message') or 'unknown error'}"
        )
        return [{"type": "error", "detail": detail}]

    messages: list[dict[str, Any]] = []
    tokens = payload.get("tokens") or []
    finals = [t for t in tokens if t.get("is_final") and t.get("text")]
    partials = [t for t in tokens if not t.get("is_final") and t.get("text")]

    if finals:
        messages.append({
            "type": "final",
            "text": "".join(str(t["text"]) for t in finals),
            "tokens": finals,
        })
    if partials:
        messages.append({
            "type": "partial",
            "text": "".join(str(t["text"]) for t in partials),
            "tokens": partials,
        })
    if payload.get("finished"):
        messages.append({
            "type": "finished",
            "final_audio_proc_ms": payload.get("final_audio_proc_ms"),
            "total_audio_proc_ms": payload.get("total_audio_proc_ms"),
        })
    return messages


def audio_filename_for_format(audio_format: str) -> str:
    """Pick a download filename extension for the buffered recording."""
    fmt = (audio_format or "auto").lower()
    if fmt.startswith("pcm") or fmt in {"mulaw", "alaw"}:
        return "realtime.pcm"
    return f"realtime{_AUDIO_EXTENSIONS.get(fmt, '.bin')}"


def duration_seconds_from_proc_ms(
    final_audio_proc_ms: int | float | None,
    total_audio_proc_ms: int | float | None,
) -> float:
    """Convert Soniox audio progress to seconds (min ~1s for metering)."""
    ms = final_audio_proc_ms or total_audio_proc_ms or 1000
    return max(float(ms), 1000.0) / 1000.0


class AudioBuffer:
    """In-memory audio buffer with a hard size ceiling."""

    def __init__(self, max_bytes: int) -> None:
        """Initialize an empty buffer capped at ``max_bytes``."""
        self._chunks: list[bytes] = []
        self._size = 0
        self.max_bytes = max_bytes

    @property
    def size(self) -> int:
        """Current buffered byte count."""
        return self._size

    def append(self, data: bytes) -> None:
        """Append a frame or raise ``AudioBufferError`` if over limit."""
        if self._size + len(data) > self.max_bytes:
            raise AudioBufferError
        self._chunks.append(data)
        self._size += len(data)

    def getvalue(self) -> bytes:
        """Return concatenated buffered audio bytes."""
        return b"".join(self._chunks)


async def persist_realtime_session(
    *,
    user: UserData,
    audio: bytes,
    audio_format: str,
    model: str | None,
    transcript: str,
    final_audio_proc_ms: int | float | None,
    total_audio_proc_ms: int | float | None,
) -> tuple[TranscribeTask | None, str | None, str | None]:
    """
    Upload recording and create a ``TranscribeTask``.

    Returns:
        (task, file_url, error_detail)
    """
    file_url: str | None = None
    if audio:
        buf = BytesIO(audio)
        buf.name = audio_filename_for_format(audio_format)
        try:
            file_url = await media.upload_file(buf)
        except Exception:
            logger.exception("Failed to upload realtime audio for user %s", user.uid)
            return None, None, "Failed to upload recorded audio"

    duration_s = duration_seconds_from_proc_ms(
        final_audio_proc_ms,
        total_audio_proc_ms,
    )
    try:
        task = await TranscribeTask.create_item({
            "file_url": file_url or "realtime://missing-audio",
            "tenant_id": user.tenant_id,
            "user_id": str(user.uid),
            "provider": "soniox",
            "model": model,
            "result": transcript or None,
            "audio_duration_seconds": duration_s,
            "task_status": (
                TaskStatusEnum.completed if file_url else TaskStatusEnum.error
            ),
            "meta_data": {
                "realtime": True,
                "audio_format": audio_format,
                "final_audio_proc_ms": final_audio_proc_ms,
                "total_audio_proc_ms": total_audio_proc_ms,
            },
            "provider_meta": {
                "provider": "soniox",
                "model": model,
                "usage": {"audio_duration_seconds": duration_s},
                "realtime": True,
            },
        })
    except Exception:
        logger.exception("Failed to persist realtime TranscribeTask")
        return None, file_url, "Failed to save transcription task"

    return task, file_url, None if file_url else "Recorded audio was empty"


async def meter_realtime_session(
    *,
    user_id: str,
    model: str | None,
    final_audio_proc_ms: int | float | None,
    total_audio_proc_ms: int | float | None,
    task_uid: str | None = None,
) -> None:
    """Meter transcription cost; log failures without failing the session."""
    duration_s = duration_seconds_from_proc_ms(
        final_audio_proc_ms,
        total_audio_proc_ms,
    )
    amount = finance.estimate_transcribe_cost(
        minutes=duration_s / 60.0,
        provider="soniox",
    )
    try:
        await finance.meter_cost(
            user_id,
            amount,
            meta_data={
                "service": "transcribe",
                "provider": "soniox",
                "model": model,
                "realtime": True,
                "task_uid": task_uid,
                "audio_duration_seconds": duration_s,
            },
        )
    except Exception:
        logger.exception("Realtime meter_cost failed for user %s", user_id)


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload))


async def handle_realtime_session(websocket: WebSocket) -> None:
    """Authenticate, bridge to Soniox RT, persist recording, and meter usage."""
    try:
        user = authenticate_websocket(websocket)
    except USSOException:
        await websocket.close(code=SONIOX_WS_CLOSE_UNAUTHORIZED)
        return

    if not Settings.soniox_api_key:
        await websocket.accept()
        await _send_json(
            websocket,
            {"type": "error", "detail": "SONIOX_API_KEY is not configured"},
        )
        await websocket.close(code=SONIOX_WS_CLOSE_FORBIDDEN)
        return

    await websocket.accept()

    try:
        await _run_bridge(websocket, user)
    except WebSocketDisconnect:
        logger.info("Realtime client disconnected user=%s", user.uid)
    except Exception:
        logger.exception("Realtime session failed user=%s", user.uid)
        try:
            await _send_json(
                websocket,
                {"type": "error", "detail": "Realtime session failed"},
            )
            await websocket.close()
        except Exception:
            logger.debug("Could not send error close to client", exc_info=True)


async def _read_client_config(websocket: WebSocket) -> dict[str, Any] | None:
    raw_config = await websocket.receive_text()
    try:
        client_config = json.loads(raw_config)
    except json.JSONDecodeError:
        await _send_json(websocket, {"type": "error", "detail": "Invalid JSON config"})
        await websocket.close()
        return None
    if not isinstance(client_config, dict):
        await _send_json(
            websocket,
            {"type": "error", "detail": "Config must be a JSON object"},
        )
        await websocket.close()
        return None
    return client_config


async def _forward_audio_frame(
    *,
    websocket: WebSocket,
    soniox_ws: ClientConnection,
    audio_buffer: AudioBuffer,
    data: bytes,
    stop: asyncio.Event,
) -> None:
    if len(data) == 0:
        await soniox_ws.send(b"")
        return
    try:
        audio_buffer.append(data)
    except AudioBufferError:
        limit = audio_buffer.max_bytes
        await _send_json(
            websocket,
            {
                "type": "error",
                "detail": f"Audio buffer exceeds limit of {limit} bytes",
            },
        )
        stop.set()
        return
    await soniox_ws.send(data)


async def _forward_control_message(
    *,
    websocket: WebSocket,
    soniox_ws: ClientConnection,
    text: str,
) -> None:
    if text == "":
        await soniox_ws.send("")
        return
    try:
        control = json.loads(text)
    except json.JSONDecodeError:
        await _send_json(
            websocket,
            {"type": "error", "detail": "Invalid control JSON"},
        )
        return
    if not isinstance(control, dict):
        return
    control_type = control.get("type")
    if control_type == "end":
        await soniox_ws.send(b"")
    elif control_type in {"finalize", "keepalive"}:
        await soniox_ws.send(json.dumps({"type": control_type}))
    else:
        await _send_json(
            websocket,
            {"type": "error", "detail": f"Unknown control: {control_type}"},
        )


def _update_progress(progress: dict[str, Any], payload: dict[str, Any]) -> None:
    if payload.get("final_audio_proc_ms") is not None:
        progress["final"] = payload["final_audio_proc_ms"]
    if payload.get("total_audio_proc_ms") is not None:
        progress["total"] = payload["total_audio_proc_ms"]


async def _emit_soniox_messages(
    *,
    websocket: WebSocket,
    payload: dict[str, Any],
    final_text_parts: list[str],
    stop: asyncio.Event,
) -> dict[str, Any] | None:
    """Forward normalized messages; return finished fields when session ends."""
    for msg in normalize_soniox_message(payload):
        if msg["type"] == "final":
            final_text_parts.append(str(msg.get("text") or ""))
        if msg["type"] == "finished":
            stop.set()
            return {
                "final_audio_proc_ms": msg.get("final_audio_proc_ms"),
                "total_audio_proc_ms": msg.get("total_audio_proc_ms"),
            }
        await _send_json(websocket, msg)
    return None


def _merge_finished(
    finished: dict[str, Any] | None,
    progress: dict[str, Any],
) -> dict[str, Any] | None:
    if finished is None:
        if progress.get("final") or progress.get("total"):
            return {
                "final_audio_proc_ms": progress.get("final"),
                "total_audio_proc_ms": progress.get("total"),
            }
        return None
    if finished.get("final_audio_proc_ms") is None:
        finished["final_audio_proc_ms"] = progress.get("final")
    if finished.get("total_audio_proc_ms") is None:
        finished["total_audio_proc_ms"] = progress.get("total")
    return finished


async def _client_to_soniox(
    *,
    websocket: WebSocket,
    soniox_ws: ClientConnection,
    audio_buffer: AudioBuffer,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            stop.set()
            return
        data = message.get("bytes")
        if data is not None:
            await _forward_audio_frame(
                websocket=websocket,
                soniox_ws=soniox_ws,
                audio_buffer=audio_buffer,
                data=data,
                stop=stop,
            )
            continue
        text = message.get("text")
        if text is not None:
            await _forward_control_message(
                websocket=websocket,
                soniox_ws=soniox_ws,
                text=text,
            )


async def _soniox_to_client(
    *,
    websocket: WebSocket,
    soniox_ws: ClientConnection,
    final_text_parts: list[str],
    progress: dict[str, Any],
    stop: asyncio.Event,
    finished_holder: dict[str, Any],
) -> None:
    try:
        async for raw in soniox_ws:
            if stop.is_set():
                break
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Non-JSON Soniox message ignored")
                continue
            if not isinstance(payload, dict):
                continue
            _update_progress(progress, payload)
            finished = await _emit_soniox_messages(
                websocket=websocket,
                payload=payload,
                final_text_parts=final_text_parts,
                stop=stop,
            )
            if finished is not None:
                finished_holder.update(finished)
                break
    finally:
        stop.set()


async def _bridge_loops(
    *,
    websocket: WebSocket,
    soniox_ws: ClientConnection,
    audio_buffer: AudioBuffer,
    final_text_parts: list[str],
    progress: dict[str, Any],
) -> dict[str, Any] | None:
    """Bidirectional proxy until Soniox finishes or client ends."""
    stop = asyncio.Event()
    finished_holder: dict[str, Any] = {}

    client_task = asyncio.create_task(
        _client_to_soniox(
            websocket=websocket,
            soniox_ws=soniox_ws,
            audio_buffer=audio_buffer,
            stop=stop,
        ),
    )
    soniox_task = asyncio.create_task(
        _soniox_to_client(
            websocket=websocket,
            soniox_ws=soniox_ws,
            final_text_parts=final_text_parts,
            progress=progress,
            stop=stop,
            finished_holder=finished_holder,
        ),
    )
    done, pending = await asyncio.wait(
        {client_task, soniox_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    stop.set()
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        exc = task.exception()
        if exc and not isinstance(exc, (asyncio.CancelledError, WebSocketDisconnect)):
            raise exc

    finished = finished_holder or None
    return _merge_finished(finished, progress)


async def _run_bridge(websocket: WebSocket, user: UserData) -> None:
    client_config = await _read_client_config(websocket)
    if client_config is None:
        return

    soniox_config = build_soniox_config(
        client_config,
        api_key=Settings.soniox_api_key or "",
        client_reference_id=str(user.uid),
    )
    audio_format = str(soniox_config.get("audio_format") or "auto")
    model = soniox_config.get("model")
    model_str = str(model) if model is not None else None

    audio_buffer = AudioBuffer(Settings.soniox_rt_max_buffer_bytes)
    final_text_parts: list[str] = []
    progress: dict[str, Any] = {"final": None, "total": None}

    async with websockets.connect(Settings.soniox_ws_url) as soniox_ws:
        await soniox_ws.send(json.dumps(soniox_config))
        finished_payload = await _bridge_loops(
            websocket=websocket,
            soniox_ws=soniox_ws,
            audio_buffer=audio_buffer,
            final_text_parts=final_text_parts,
            progress=progress,
        )

    finished = finished_payload or {}
    progress_final = finished.get("final_audio_proc_ms", progress["final"])
    progress_total = finished.get("total_audio_proc_ms", progress["total"])

    task, file_url, persist_error = await persist_realtime_session(
        user=user,
        audio=audio_buffer.getvalue(),
        audio_format=audio_format,
        model=model_str,
        transcript="".join(final_text_parts),
        final_audio_proc_ms=progress_final,
        total_audio_proc_ms=progress_total,
    )

    await meter_realtime_session(
        user_id=str(user.uid),
        model=model_str,
        final_audio_proc_ms=progress_final,
        total_audio_proc_ms=progress_total,
        task_uid=str(task.uid) if task is not None else None,
    )

    if persist_error:
        await _send_json(websocket, {"type": "error", "detail": persist_error})

    finished_msg: dict[str, Any] = {
        "type": "finished",
        "final_audio_proc_ms": progress_final,
        "total_audio_proc_ms": progress_total,
    }
    if task is not None:
        finished_msg["task_uid"] = str(task.uid)
    if file_url:
        finished_msg["file_url"] = file_url
    await _send_json(websocket, finished_msg)
    await websocket.close()
