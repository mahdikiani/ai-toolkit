"""OpenAI-compatible audio endpoints backed by OpenRouter TTS + Soniox."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import suppress
from pathlib import Path

import httpx
from fastapi.responses import Response
from soniox.languages import Language
from soniox.types import TranscriptionConfig, TranscriptionJobStatus

from apps.openai_compat.services import openai_error
from apps.transcribe import services as transcribe_services
from server.config import Settings
from utils.billing import finance
from utils.integrations.openrouter import build_headers, resolve_api_key

logger = logging.getLogger(__name__)


def _estimate_speech_cost(text: str) -> float:
    """Estimate TTS cost from character count."""
    pricing = finance.pricing_config().get("speech") or {}
    per_1k = float(pricing.get("default_per_1k_chars", 0.5))
    markup = float(pricing.get("markup", 1.0))
    chars = max(1, len(text))
    return (chars / 1000) * per_1k * markup


def _language_hints(language: str | None) -> list:
    """Build Soniox language hints from an optional language code."""
    hints = [Language.PERSIAN, Language.ENGLISH]
    if not language:
        return hints
    with suppress(Exception):
        return [Language(language)]
    return hints


async def _poll_transcription(job_id: str) -> object:
    """Poll Soniox until the job completes or fails."""
    soniox = transcribe_services.get_soniox_client()
    poll_interval = Settings.transcribe_poll_interval_seconds
    for _ in range(120):
        job_result = await soniox.get_transcription_job_async(job_id)
        if job_result.status == TranscriptionJobStatus.COMPLETED:
            return job_result
        if job_result.status == TranscriptionJobStatus.ERROR:
            raise openai_error(502, "upstream_error", "Transcription job failed")
        await asyncio.sleep(poll_interval)
    raise openai_error(504, "timeout", "Transcription timed out")


async def create_speech(body: dict, *, user_id: str) -> Response:
    """Proxy OpenAI-style TTS to OpenRouter `/audio/speech`."""
    text = body.get("input") or body.get("text")
    if not text or not isinstance(text, str):
        raise openai_error(400, "invalid_request_error", "input is required")

    model = body.get("model") or getattr(
        Settings, "default_tts_model", "openai/gpt-4o-mini-tts"
    )
    voice = body.get("voice", "alloy")
    response_format = body.get("response_format", "mp3")

    estimated = _estimate_speech_cost(text)
    await finance.check_quota(user_id, estimated, raise_exception=True)

    try:
        resolve_api_key()
    except Exception as exc:
        raise openai_error(503, "service_unavailable", str(exc)) from exc

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
    }
    url = f"{Settings.openrouter_base_url.rstrip('/')}/audio/speech"
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload, headers=build_headers())

    if resp.status_code >= 400:
        raise openai_error(resp.status_code, "upstream_error", resp.text[:500])

    try:
        await finance.meter_cost(
            user_id,
            estimated,
            meta_data={
                "service": "openai_compat_speech",
                "model": model,
                "chars": len(text),
            },
        )
    except Exception:
        logger.exception("Failed to meter openai_compat speech usage")

    media = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(str(response_format), "application/octet-stream")
    return Response(content=resp.content, media_type=media)


async def create_transcription(
    content: bytes,
    *,
    filename: str,
    user_id: str,
    model: str | None = None,
    language: str | None = None,
    response_format: str = "json",
) -> dict:
    """Transcribe uploaded audio via Soniox and return OpenAI-shaped JSON."""
    if not content:
        raise openai_error(400, "invalid_request_error", "file is required")
    if not Settings.soniox_api_key:
        raise openai_error(
            503, "service_unavailable", "SONIOX_API_KEY is not configured"
        )

    estimated = finance.estimate_transcribe_cost(minutes=1.0, provider="soniox")
    await finance.check_quota(user_id, estimated, raise_exception=True)

    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()

        config = TranscriptionConfig(
            language_hints=_language_hints(language),
            enable_language_identification=True,
            enable_speaker_diarization=False,
            client_reference_id=f"openai-compat:{user_id}",
        )
        soniox = transcribe_services.get_soniox_client()
        job = await soniox.transcribe_file_async(tmp.name, config)
        job_result = await _poll_transcription(job.id)
        result = await soniox.get_transcription_result_async(job.id)

        text = getattr(result, "text", None) or ""
        if not text and hasattr(result, "tokens"):
            text = "".join(getattr(tok, "text", "") for tok in (result.tokens or []))

        duration_ms = getattr(job_result, "audio_duration_ms", None) or 0
        minutes = max(duration_ms / 60_000, 1 / 60)
        amount = finance.estimate_transcribe_cost(minutes=minutes, provider="soniox")
        try:
            await finance.meter_cost(
                user_id,
                amount,
                meta_data={
                    "service": "openai_compat_transcriptions",
                    "model": model or "soniox",
                    "minutes": minutes,
                },
            )
        except Exception:
            logger.exception("Failed to meter openai_compat transcription usage")

    fmt = (response_format or "json").lower()
    if fmt == "text":
        return {"text": text}
    if fmt == "verbose_json":
        return {
            "task": "transcribe",
            "language": language,
            "duration": duration_ms / 1000 if duration_ms else None,
            "text": text,
        }
    return {"text": text}
