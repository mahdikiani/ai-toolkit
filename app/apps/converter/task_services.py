"""Process ConversionTask created via from-media."""

from __future__ import annotations

import logging

from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.artifacts.enums import EXTENSION_BY_FORMAT, MIME_BY_FORMAT
from apps.artifacts.services import create_artifact_from_bytes
from utils.integrations.media import download_by_storage_uri

from . import registry
from .media_source import normalize_media_source_uri
from .services import convert_artifact
from .task_models import ConversionTask

logger = logging.getLogger(__name__)


async def process_conversion_from_media(task: ConversionTask) -> ConversionTask:
    """Ingest Media URI, convert Artifacts, then emit webhook status."""
    try:
        await task.save_status(TaskStatusEnum.processing)

        registry.ensure_builtin_strategies()
        if registry.get_edge(task.source_format, task.target_format) is None:
            await task.update_and_emit(
                task_status=TaskStatusEnum.error,
                task_report=(
                    f"unsupported_conversion:"
                    f"{task.source_format.value}->{task.target_format.value}"
                ),
            )
            return task

        storage_uri = normalize_media_source_uri(task.source_uri)
        raw = await download_by_storage_uri(storage_uri)

        filename = (
            task.original_name
            or f"{(task.title or 'document').strip() or 'document'}"
            f"{EXTENSION_BY_FORMAT[task.source_format]}"
        )
        source_artifact = await create_artifact_from_bytes(
            data=raw,
            filename=filename,
            content_type=MIME_BY_FORMAT[task.source_format],
            artifact_format=task.source_format,
            user_id=task.user_id,
            tenant_id=task.tenant_id,
            workspace_id=task.workspace_id,
            title=task.title,
            original_name=task.original_name or filename,
            source="conversion_from_media",
            meta_data={"conversion_task_uid": task.uid, "source_uri": storage_uri},
        )

        derived = await convert_artifact(
            artifact_id=str(source_artifact.uid),
            target_format=task.target_format,
            user_id=task.user_id,
            tenant_id=task.tenant_id,
            workspace_id=task.workspace_id,
        )

        task.source_artifact_id = str(source_artifact.uid)
        task.result_artifact_id = str(derived.uid)
        task.result_storage_uri = derived.storage_uri
        task.result = {
            "source_artifact_id": task.source_artifact_id,
            "result_artifact_id": task.result_artifact_id,
            "result_storage_uri": task.result_storage_uri,
            "source_format": task.source_format.value,
            "target_format": task.target_format.value,
        }
        task.provider_meta = {
            "pipeline": "conversions_from_media_v1",
            "source_uri": storage_uri,
        }
        await task.update_and_emit(
            task_status=TaskStatusEnum.completed,
            task_report="Task processed successfully",
        )
        logger.info(
            "convert.manual_verify conversion_task uid=%s source_artifact=%s "
            "result_artifact=%s target=%s",
            task.uid,
            task.source_artifact_id,
            task.result_artifact_id,
            task.target_format.value,
        )
    except Exception as exc:
        logger.exception("ConversionTask %s failed", task.uid)
        await task.update_and_emit(
            task_status=TaskStatusEnum.error,
            task_report=str(exc)[:500],
        )
    return task
