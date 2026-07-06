"""Translation via the shared translate prompt + OpenRouter."""

from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.language.promptic.engine import PromptEngine, load_data
from apps.language.promptic.services import call_openrouter
from server.config import Settings
from utils import finance, texttools

from .models import TranslateTask

TRANSLATE_PROMPT = "translate.yaml"


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
        meta = load_data(prompt_path)
        if not isinstance(meta, dict):
            raise TypeError("translate prompt must be a YAML mapping")

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
        amount = finance.estimate_text_cost(
            model=str(provider_meta.get("model") or model or ""),
            usage=provider_usage if isinstance(provider_usage, dict) else None,
            raw_cost=provider_meta.get("raw_cost"),
        )
        usage = await finance.meter_cost(
            task.user_id,
            amount,
            meta_data={
                "service": "translate",
                "prompt": "translate",
                "provider_meta": provider_meta,
            },
        )
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
