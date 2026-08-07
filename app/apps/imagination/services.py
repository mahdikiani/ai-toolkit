"""Imagination services — prompt enhancement + image generation via OpenRouter."""

import logging

import httpx
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils.billing import finance
from utils.integrations.openrouter import build_headers, resolve_api_key

from .models import ImaginationTask

logger = logging.getLogger(__name__)


ENHANCE_SYSTEM_PROMPT = (
    "You are a creative prompt engineer. Your task is to expand the user's short "
    "imagination prompt into a detailed, vivid image generation prompt suitable "
    "for models like DALL-E or Stable Diffusion. Return ONLY the enhanced prompt "
    "text, no explanations, no markdown, no quotes."
)


async def _enhance_prompt(prompt: str) -> str:
    """Enrich a short prompt into a detailed image prompt via an LLM."""
    model = Settings.default_model
    url = f"{Settings.openrouter_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": ENHANCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 300,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=build_headers())
            resp.raise_for_status()
            data = resp.json()
            enhanced = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            return enhanced or prompt
    except Exception as exc:
        logger.warning("Prompt enhancement failed, using original: %s", exc)
        return prompt


async def _generate_image(prompt: str, task: ImaginationTask) -> dict:
    """Generate an image via OpenRouter's images/generations endpoint."""
    model = task.model or "openai/dall-e-3"
    url = f"{Settings.openrouter_base_url.rstrip('/')}/images/generations"
    payload = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": task.size,
        "response_format": "url",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload, headers=build_headers())
        resp.raise_for_status()
        return resp.json()


async def process_imagination(task: ImaginationTask) -> ImaginationTask:
    """Generate an image for a task, optionally enhancing the prompt first."""
    try:
        resolve_api_key()
    except Exception as exc:
        task.task_status = TaskStatusEnum.error
        task.result_url = None
        await task.save_report(str(exc))
        return task

    estimated = _estimate_imagination_cost(task)
    quota = await finance.check_quota(
        task.user_id, estimated, raise_exception=False, workspace_id=task.workspace_id
    )
    if quota < estimated:
        task.task_status = TaskStatusEnum.error
        task.result_url = None
        await task.save_report("insufficient_quota")
        return task

    if task.enhance_prompt:
        try:
            task.enhanced_prompt = await _enhance_prompt(task.prompt)
        except Exception as exc:
            logger.warning("Enhancement failed, using original: %s", exc)
            task.enhanced_prompt = task.prompt
    else:
        task.enhanced_prompt = task.prompt

    final_prompt = task.enhanced_prompt or task.prompt

    try:
        result = await _generate_image(final_prompt, task)
    except httpx.HTTPStatusError as exc:
        task.task_status = TaskStatusEnum.error
        error_detail = exc.response.text[:500]
        await task.save_report(f"Image generation error: {error_detail}")
        return task
    except Exception as exc:
        task.task_status = TaskStatusEnum.error
        await task.save_report(f"Image generation failed: {exc}")
        return task

    data_list = result.get("data", [])
    if not data_list:
        task.task_status = TaskStatusEnum.error
        await task.save_report("No image data returned")
        return task

    image_data = data_list[0]
    result_url = image_data.get("url")
    result_b64 = image_data.get("b64_json")

    task.result_url = result_url
    task.result_b64 = result_b64

    usage = None
    try:
        usage = await finance.meter_cost(
            task.user_id,
            estimated,
            meta_data={
                "service": "imagination",
                "model": task.model or "openai/dall-e-3",
                "enhanced": task.enhance_prompt,
                "task_uid": task.uid,
            },
            workspace_id=task.workspace_id,
        )
    except Exception:
        logger.exception("Failed to meter imagination usage")

    task.usage_amount = float(usage.amount) if usage else estimated
    task.usage_id = usage.uid if usage else None

    task.task_status = TaskStatusEnum.completed
    await task.save_report("Image generated successfully")
    return task


def _estimate_imagination_cost(task: ImaginationTask) -> float:
    return finance.estimate_image_cost(model=task.model)
