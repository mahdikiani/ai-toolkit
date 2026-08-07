"""Text-to-speech via OpenRouter's /audio/speech endpoint."""

import logging

import httpx
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils.billing import finance
from utils.integrations.openrouter import build_headers, resolve_api_key

from .models import TextToSpeechTask

logger = logging.getLogger(__name__)


async def process_tts(task: TextToSpeechTask) -> TextToSpeechTask:
    """Generate speech via OpenRouter, gated and billed by character count."""
    try:
        resolve_api_key()
    except Exception as exc:
        task.task_status = TaskStatusEnum.error
        await task.save_report(str(exc))
        return task

    amount = finance.estimate_speech_cost(chars=len(task.text))

    quota = await finance.check_quota(
        task.user_id, amount, raise_exception=False, workspace_id=task.workspace_id
    )
    if quota < amount:
        task.task_status = TaskStatusEnum.error
        await task.save_report("insufficient_quota")
        return task

    payload = {
        "model": task.model,
        "input": task.text,
        "voice": task.voice,
        "response_format": task.response_format,
        "speed": task.speed,
    }
    url = f"{Settings.openrouter_base_url.rstrip('/')}/audio/speech"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=build_headers())
            resp.raise_for_status()
            audio_bytes = resp.content
    except Exception as exc:
        task.task_status = TaskStatusEnum.error
        await task.save_report(f"TTS failed: {exc}")
        return task

    task.result_data = audio_bytes

    usage = None
    try:
        usage = await finance.meter_cost(
            task.user_id,
            amount,
            meta_data={
                "service": "texttospeech",
                "provider": "openrouter",
                "model": task.model,
                "chars": len(task.text),
                "task_uid": task.uid,
            },
            workspace_id=task.workspace_id,
        )
    except Exception:
        logger.exception("Failed to meter TTS usage")

    task.usage_amount = float(usage.amount) if usage else amount
    task.usage_id = usage.uid if usage else None

    task.task_status = TaskStatusEnum.completed
    await task.save_report("Speech generated successfully")
    return task
