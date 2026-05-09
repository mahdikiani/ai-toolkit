"""Beanie models for chat."""

from typing import ClassVar

from fastapi_mongo_base.models import UserOwnedEntity
from pymongo import ASCENDING, IndexModel

from .schemas import ChatMessageSchema, ChatSessionSchema, ChatThreadSchema


class ChatSession(UserOwnedEntity, ChatSessionSchema):  # type: ignore[misc]
    """Conversation container; threads isolate model/branches."""


class ChatThread(UserOwnedEntity, ChatThreadSchema):  # type: ignore[misc]
    """One branch inside a session."""

    class Settings:
        """Database collection settings for chat threads."""

        indexes: ClassVar[list[IndexModel]] = [
            *UserOwnedEntity.Settings.indexes,
            IndexModel([("session_uid", ASCENDING)]),
        ]


class ChatMessage(UserOwnedEntity, ChatMessageSchema):  # type: ignore[misc]
    """Single message bound to a thread."""

    class Settings:
        """Database collection settings for chat messages."""

        indexes: ClassVar[list[IndexModel]] = [
            *UserOwnedEntity.Settings.indexes,
            IndexModel([("thread_uid", ASCENDING), ("created_at", ASCENDING)]),
        ]
