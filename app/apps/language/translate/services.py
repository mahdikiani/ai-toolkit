"""Translation via the shared translate prompt + OpenRouter."""

import logging
from pathlib import Path

from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.language.promptic.engine import PromptEngine, load_data
from apps.language.promptic.services import call_openrouter
from server.config import Settings
from utils import texttools
from utils.billing import finance

from .models import TranslateTask

logger = logging.getLogger(__name__)

TRANSLATE_PROMPT = "translate.yaml"


class TranslatePromptFormatError(TypeError):
    """Raised when the translate prompt does not contain a YAML mapping."""


def _text_cost_value(value: object) -> int | float | str | None:
    """Return a provider cost value accepted by the finance service."""
    if isinstance(value, (int, float, str)):
        return value
    return None


def _load_translate_metadata(prompt_path: Path) -> dict:
    """Load and validate the translation prompt metadata."""
    meta = load_data(prompt_path)
    if not isinstance(meta, dict):
        error = TranslatePromptFormatError("translate prompt must be a YAML mapping")
        raise error
    return meta


async def process_translate(task: TranslateTask) -> TranslateTask:
    """Run `executions/prompts/translate.yaml` through PromptEngine and OpenRouter."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / TRANSLATE_PROMPT

    if not prompt_path.exists():
        task.error = f"Prompt '{TRANSLATE_PROMPT}' not found"
        task.task_status = TaskStatusEnum.error
        await task.save()
        return task

    try:
        meta = _load_translate_metadata(prompt_path)

        engine = PromptEngine(base_dir=prompts_dir)
        input_variables = {
            "content": task.text,
            "language": task.language or "Persian",
        }
        system_prompt, user_prompt, response_format = engine.generate(
            prompt_path,
            input_variables,
        )

        model = meta.get("model")
        temperature = float(meta.get("temperature", 0.2))
        max_tokens = meta.get("max_tokens")
        if max_tokens is not None:
            max_tokens = int(max_tokens)

        estimated = finance.estimate_text_cost(
            model=str(model or ""),
            usage={"total_tokens": max(100, len(task.text) // 4) * 2},
        )
        quota = await finance.check_quota(
            task.user_id,
            estimated,
            raise_exception=False,
            workspace_id=task.workspace_id,
        )
        if quota < estimated:
            task.error = "insufficient_quota"
            task.task_status = TaskStatusEnum.error
            await task.save()
            return task

        openrouter_result = await call_openrouter(
            system_prompt,
            user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            return_meta=True,
        )
        if isinstance(openrouter_result, tuple):
            result, provider_meta = openrouter_result
        else:
            result = openrouter_result
            provider_meta = {}

        provider_usage = provider_meta.get("usage") if provider_meta else None
        raw_cost = provider_meta.get("raw_cost")
        amount = finance.estimate_text_cost(
            model=str(provider_meta.get("model") or model or ""),
            usage=provider_usage if isinstance(provider_usage, dict) else None,
            raw_cost=_text_cost_value(raw_cost),
        )
        usage = None
        try:
            usage = await finance.meter_cost(
                task.user_id,
                amount,
                meta_data={
                    "service": "translate",
                    "prompt": "translate",
                    "provider_meta": provider_meta,
                },
                workspace_id=task.workspace_id,
            )
        except Exception:
            logger.exception("Failed to meter translate usage for task %s", task.uid)
        await save_result(
            task,
            result,
            provider_meta=provider_meta,
            usage_amount=float(usage.amount) if usage else None,
            usage_id=usage.uid if usage else None,
        )

    except Exception as e:
        task.error = str(e)
        task.task_status = TaskStatusEnum.error
        await task.save()

    return task


async def save_result(
    task: TranslateTask,
    result: str,
    provider_meta: dict | None = None,
    usage_amount: float | None = None,
    usage_id: str | None = None,
) -> TranslateTask:
    """Save successful result for a translation task."""
    task.result = texttools.normalize_text(result)
    task.provider_meta = provider_meta
    task.task_status = TaskStatusEnum.completed
    task.usage_amount = usage_amount
    task.usage_id = usage_id
    return await task.save()
