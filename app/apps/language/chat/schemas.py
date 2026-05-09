"""Pydantic schemas for chat sessions, threads, and messages."""

from typing import Literal

from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from pydantic import BaseModel, Field


class ChatSessionCreate(BaseModel):
    """Create a chat session (first thread is created automatically)."""

    title: str | None = Field(None, description="Optional session title")
    initial_thread_title: str | None = Field(
        None,
        description="Title for the first thread (default: Thread 1)",
    )
    initial_chat_model: str | None = Field(
        None,
        description="OpenRouter model id for the first thread",
    )


class ChatSessionUpdate(BaseModel):
    """Patch session metadata."""

    title: str | None = None
    active_thread_uid: str | None = None


class ChatSessionSchema(UserOwnedEntitySchema):
    """Stored chat session."""

    title: str | None = None
    active_thread_uid: str | None = Field(
        None,
        description="Hint for UI: last-focused thread uid",
    )


class ChatThreadCreate(BaseModel):
    """Create another thread inside a session (different model / branch)."""

    title: str | None = Field(None, description="Optional thread title")
    chat_model: str | None = Field(
        None,
        description="OpenRouter model id; falls back to server default when omitted",
    )


class ChatThreadSchema(UserOwnedEntitySchema):
    """Stored chat thread."""

    session_uid: str = Field(..., description="Owning session uid")
    title: str | None = None
    chat_model: str | None = Field(
        None,
        description="Model used for completions in this thread",
    )


class ChatMessageCreate(BaseModel):
    """Append a message; optionally request an assistant reply."""

    role: Literal["user", "assistant", "system"] = Field(
        default="user",
        description="Usually user for new turns",
    )
    content: str = Field(..., min_length=1)
    reply_to_uid: str | None = Field(
        None,
        description="Optional reference to another message uid (UI threading)",
    )
    generate_reply: bool = Field(
        False,
        description=(
            "If true, call OpenRouter with full thread history "
            "and append assistant message"
        ),
    )
    stream: bool = Field(
        False,
        description="When generate_reply is true, stream SSE chunks like OpenRouter",
    )


class ChatMessageSchema(UserOwnedEntitySchema):
    """Stored chat message."""

    thread_uid: str
    role: Literal["user", "assistant", "system"]
    content: str
    reply_to_uid: str | None = None
    completion_extra: dict | None = Field(
        None,
        description="Optional usage / routing metadata after assistant generation",
    )


class ChatCompletionResponse(BaseModel):
    """Non-streaming reply to POST .../messages when generate_reply is true."""

    user_message: ChatMessageSchema
    assistant_message: ChatMessageSchema | None = None
