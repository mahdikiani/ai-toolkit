"""Text-to-speech task model."""

from fastapi_mongo_base.models import TenantUserEntity
from fastapi_mongo_base.tasks import TaskStatusEnum

from .schemas import TextToSpeechTaskSchema


class TextToSpeechTask(TenantUserEntity, TextToSpeechTaskSchema):
    """Text-to-speech task entity."""

    async def start_processing(self) -> "TextToSpeechTask":
        """Start processing the text-to-speech task."""
        from . import services

        self.task_status = TaskStatusEnum.processing
        return await services.process_tts(self)
