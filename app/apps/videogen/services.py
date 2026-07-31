"""Video generation via OpenRouter chat completions with video models."""

import json
import logging

import httpx
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils.billing import finance
from utils.integrations.openrouter import build_headers, resolve_api_key
from utils.integrations.replicate import create_prediction

from .models import VideoGenTask

logger = logging.getLogger(__name__)


async def _generate_via_openrouter(task: VideoGenTask) -> str:
    messages = [
        {
            "role": "user",
            "content": [],
        }
    ]
    if task.image_url:
        messages[0]["content"].append(
            {"type": "image_url", "image_url": {"url": task.image_url}}
        )
    messages[0]["content"].append({"type": "text", "text": task.prompt})

    payload = {
        "model": task.model,
        "messages": messages,
        "max_tokens": 2048,
    }
    url = f"{Settings.openrouter_base_url.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, json=payload, headers=build_headers())
        resp.raise_for_status()
        data = resp.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    try:
        parsed = json.loads(content)
        video_url = parsed.get("video_url") or parsed.get("url") or content
    except (json.JSONDecodeError, TypeError):
        video_url = content

    return str(video_url).strip() if video_url else content


async def _generate_via_replicate(task: VideoGenTask) -> str:
    result = await create_prediction(
        task.model,
        {"prompt": task.prompt},
        timeout_secs=600.0,
    )
    output = result.get("output")
    if isinstance(output, list):
        return str(output[0]) if output else ""
    return str(output or "")


async def process_video(task: VideoGenTask) -> VideoGenTask:
    """Generate a video via the configured provider, gated and billed per video."""
    provider = getattr(task, "provider", "openrouter") or "openrouter"

    pricing = finance.pricing_config().get("video") or {}
    amount = float(pricing.get("default_per_video", 1.0))
    quota = await finance.check_quota(task.user_id, amount, raise_exception=False)
    if quota < amount:
        task.task_status = TaskStatusEnum.error
        await task.save_report("insufficient_quota")
        return task

    try:
        if provider == "replicate":
            video_url = await _generate_via_replicate(task)
        else:
            resolve_api_key()
            video_url = await _generate_via_openrouter(task)
    except Exception as exc:
        task.task_status = TaskStatusEnum.error
        await task.save_report(f"Video generation failed: {exc}")
        return task

    if not video_url:
        task.task_status = TaskStatusEnum.error
        await task.save_report("No video URL returned")
        return task

    task.result_url = video_url

    try:
        await finance.meter_cost(
            task.user_id,
            amount,
            meta_data={
                "service": "videogen",
                "provider": provider,
                "model": task.model,
            },
        )
    except Exception:
        logger.exception("Failed to meter video generation usage")

    task.task_status = TaskStatusEnum.completed
    await task.save_report("Video generated successfully")
    return task
