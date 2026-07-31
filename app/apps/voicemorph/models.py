"""Voice morphing task model."""

from fastapi_mongo_base.models import TenantUserEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import VoiceMorphTaskSchema


class VoiceMorphTask(TenantUserEntity, VoiceMorphTaskSchema):
    """Voice morphing task entity."""

    async def start_processing(self) -> "VoiceMorphTask":
        """Start processing the voice morph task."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_voice_morph(self)
