"""Voice morphing via Replicate (So-VITS / RVC-style voice conversion)."""

import logging

from fastapi_mongo_base.tasks import TaskStatusEnum

from utils.billing import finance
from utils.integrations.replicate import create_prediction

from .models import VoiceMorphTask

logger = logging.getLogger(__name__)


REPLICATE_VOICE_MODELS = {
    "so-vits-svc": "cjwbw/so-vits-svc-5:8b8e1e2d8c3c3b4d5e6f7a8b9c0d1e2f",
}


async def process_voice_morph(task: VoiceMorphTask) -> VoiceMorphTask:
    """Morph voice audio for a task via Replicate, gated and billed per request."""
    model_key = "so-vits-svc"
    model_id = REPLICATE_VOICE_MODELS.get(model_key, "cjwbw/so-vits-svc-5")

    pricing = finance.pricing_config().get("voice_morph") or {}
    amount = float(pricing.get("default_per_request", 1.0))
    quota = await finance.check_quota(
        task.user_id, amount, raise_exception=False, workspace_id=task.workspace_id
    )
    if quota < amount:
        task.task_status = TaskStatusEnum.error
        await task.save_report("insufficient_quota")
        return task

    input_data = {
        "audio_url": task.audio_url,
    }
    if task.voice_reference_url:
        input_data["voice_url"] = task.voice_reference_url
    if task.pitch_shift is not None:
        input_data["pitch_shift"] = task.pitch_shift
    if task.speed_factor is not None:
        input_data["speed_factor"] = task.speed_factor

    try:
        result = await create_prediction(model_id, input_data, timeout_secs=300.0)
    except Exception as exc:
        task.task_status = TaskStatusEnum.error
        await task.save_report(f"Voice morph failed: {exc}")
        return task

    output = result.get("output")
    if isinstance(output, list):
        task.result_url = str(output[0]) if output else None
    elif isinstance(output, str):
        task.result_url = output
    else:
        task.result_url = str(output or "")

    if not task.result_url:
        task.task_status = TaskStatusEnum.error
        await task.save_report("No output from voice morph")
        return task

    try:
        await finance.meter_cost(
            task.user_id,
            amount,
            meta_data={
                "service": "voicemorph",
                "provider": "replicate",
                "model": model_key,
            },
            workspace_id=task.workspace_id,
        )
    except Exception:
        logger.exception("Failed to meter voice morph usage")

    task.task_status = TaskStatusEnum.completed
    await task.save_report("Voice morphed successfully")
    return task
