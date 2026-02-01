import asyncio

from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils import dify, finance, texttools

from .models import TranslateTask


async def process_translate(task: TranslateTask) -> TranslateTask:
    pages = [task.text]
    async with dify.AsyncDifyClient(Settings.pishrun_api_key) as client:
        sem = asyncio.Semaphore(16)

        async def sem_translate(page: str) -> str | None:
            async with sem:
                return await client.translate(page)

        text_pages = await asyncio.gather(*(sem_translate(page) for page in pages))

    usage = await finance.meter_cost(task.user_id, len(text_pages))
    await save_result(
        task,
        "\n\n".join([t for t in text_pages if t]),
        usage_amount=float(usage.amount) if usage else None,
        usage_id=usage.uid if usage else None,
    )

    return task


async def save_result(
    task: TranslateTask,
    result: str,
    usage_amount: float | None = None,
    usage_id: str | None = None,
) -> TranslateTask:
    task.result = texttools.normalize_text(result)
    task.task_status = TaskStatusEnum.completed
    task.usage_amount = usage_amount
    task.usage_id = usage_id
    return await task.save()
