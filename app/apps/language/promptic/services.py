"""Services for execution task management."""

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils import openrouter as openrouter_client

from .engine import PromptEngine

if TYPE_CHECKING:
    from .models import ExecutionTask
    from .schemas import ExecutionTaskCreate

logger = logging.getLogger(__name__)


def check_schemas(prompt_name: str, data: "ExecutionTaskCreate") -> None:
    """Validate that the prompt exists and input variables match schema."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{prompt_name}.yaml"

    if not prompt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prompt '{prompt_name}' not found",
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
) -> str:
    """Call OpenRouter API."""
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
    return content.strip()


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


async def invoke_stream(task: "ExecutionTask") -> AsyncIterator[str]:
    """Execute prompt and stream the response."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{task.prompt_name}.yaml"

    if not prompt_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prompt '{task.prompt_name}' not found",
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


async def process_execution_task(
    task: "ExecutionTask",
    *,
    force_restart: bool = False,
    sync: bool = False,
    **kwargs: object,
) -> "ExecutionTask":
    """Process an execution task by invoking the prompt template."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{task.prompt_name}.yaml"

    if not prompt_path.exists():
        task.error = f"Prompt '{task.prompt_name}' not found"
        task.task_status = TaskStatusEnum.error
        await task.save()
        return task

    try:
        engine = PromptEngine(base_dir=prompts_dir)
        system_prompt, user_prompt, response_format = engine.generate(
            prompt_path, task.input_variables
        )

        result = await call_openrouter(
            system_prompt,
            user_prompt,
            response_format=response_format,
            temperature=0.2,
        )

        task.result = result
        task.task_status = TaskStatusEnum.completed
        await task.save()

    except Exception as e:
        logger.exception("Error executing prompt")
        task.error = str(e)
        task.task_status = TaskStatusEnum.error
        await task.save()

    return task
