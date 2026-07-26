"""Services for promptic task management."""

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi_mongo_base.core.exceptions import BaseHTTPException
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils.billing import finance
from utils.integrations import openrouter as openrouter_client

from .engine import PromptEngine

if TYPE_CHECKING:
    from .models import PrompticTask
    from .schemas import PrompticCreate

logger = logging.getLogger(__name__)


def check_schemas(prompt_name: str, data: "PrompticCreate") -> None:
    """Validate that the prompt exists and input variables match schema."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{prompt_name}.yaml"

    if not prompt_path.exists():
        raise BaseHTTPException(
            status_code=404,
            error_code="prompt_not_found",
            detail=f"Prompt '{prompt_name}' not found",
            message={
                "en": f"Prompt '{prompt_name}' not found",
                "fa": f"پرامپت '{prompt_name}' یافت نشد",
            },
        )


async def call_openrouter(
    system: str,
    user: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    response_format: dict | None = None,
    return_meta: bool = False,
) -> str | tuple[str, dict[str, object]]:
    """Call OpenRouter API and optionally return provider metadata."""
    model = model or Settings.default_model
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    if response_format:
        body["response_format"] = response_format

    data = await openrouter_client.complete_chat_json(body, api_key=api_key)

    choices = data.get("choices")
    if not choices:
        raise RuntimeError(
            "No response from model; raw response: "
            + json.dumps(data, ensure_ascii=False)[:500]
        )
    content = choices[0].get("message", {}).get("content") or ""
    provider_meta = openrouter_client.extract_provider_meta(data, provider="openrouter")
    if not return_meta:
        return content.strip()
    return content.strip(), provider_meta


async def call_openrouter_stream(
    system: str,
    user: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
) -> AsyncIterator[str]:
    """Call OpenRouter API with streaming."""
    model = model or Settings.default_model
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens

    async for delta in openrouter_client.stream_chat_deltas(body, api_key=api_key):
        yield delta


async def invoke_stream(task: "PrompticTask") -> AsyncIterator[str]:
    """Execute prompt and stream the response."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{task.prompt_name}.yaml"

    if not prompt_path.exists():
        raise BaseHTTPException(
            status_code=404,
            error_code="prompt_not_found",
            detail=f"Prompt '{task.prompt_name}' not found",
            message={
                "en": f"Prompt '{task.prompt_name}' not found",
                "fa": f"پرامپت '{task.prompt_name}' یافت نشد",
            },
        )

    try:
        engine = PromptEngine(base_dir=prompts_dir)
        system_prompt, user_prompt, _ = engine.generate(
            prompt_path, task.input_variables
        )

        full_response = ""
        async for chunk in call_openrouter_stream(
            system_prompt,
            user_prompt,
            temperature=0.2,
        ):
            full_response += chunk
            yield chunk

        # Update task with final result
        task.result = full_response
        task.task_status = TaskStatusEnum.completed
        await task.save()

    except Exception as e:
        logger.exception("Error executing prompt")
        task.error = str(e)
        task.task_status = TaskStatusEnum.error
        await task.save()
        raise


async def process_promptic(
    task: "PrompticTask",
    *,
    force_restart: bool = False,
    sync: bool = False,
    **kwargs: object,
) -> "PrompticTask":
    """Process a promptic run by invoking the prompt template."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{task.prompt_name}.yaml"

    if not prompt_path.exists():
        await task.update_and_emit(
            error=f"Prompt '{task.prompt_name}' not found",
            task_status=TaskStatusEnum.error,
        )
        return task

    try:
        engine = PromptEngine(base_dir=prompts_dir)
        system_prompt, user_prompt, response_format = engine.generate(
            prompt_path, task.input_variables
        )

        openrouter_result = await call_openrouter(
            system_prompt,
            user_prompt,
            response_format=response_format,
            temperature=0.2,
            return_meta=True,
        )
        if isinstance(openrouter_result, tuple):
            result, provider_meta = openrouter_result
        else:
            result = openrouter_result
            provider_meta = {}

        usage = (provider_meta.get("usage") if provider_meta else None) or {}
        raw_cost = provider_meta.get("raw_cost")
        amount = finance.estimate_text_cost(
            model=str(provider_meta.get("model") or ""),
            usage=usage if isinstance(usage, dict) else None,
            raw_cost=_text_cost_value(raw_cost),
        )
        metered = await finance.meter_cost(
            task.user_id,
            amount,
            meta_data={
                "service": "promptic",
                "prompt": task.prompt_name,
                "provider_meta": provider_meta,
            },
        )

        await task.update_and_emit(
            result=result,
            provider_meta=provider_meta,
            usage_amount=float(metered.amount) if metered else amount,
            usage_id=metered.uid if metered else None,
            task_status=TaskStatusEnum.completed,
        )

    except Exception as e:
        logger.exception("Error executing prompt")
        await task.update_and_emit(
            error=str(e),
            task_status=TaskStatusEnum.error,
        )

    return task


# Backward-compatible alias used by older unit tests.
process_execution_task = process_promptic


def _text_cost_value(value: object) -> int | float | str | None:
    """Return a provider cost value accepted by the finance service."""
    if isinstance(value, (int, float, str)):
        return value
    return None
