"""Pydantic schemas for chat sessions, threads, and messages."""

from typing import Literal

from fastapi_mongo_base.schemas import UserOwnedEntitySchema
from pydantic import BaseModel, Field

from utils.workspace import WorkspaceScopedSchema


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
    active_thread_uid: str | None = Field(
        None,
        description="Hint for UI: last-focused thread uid",
    )


class ChatSessionSchema(
    UserOwnedEntitySchema, WorkspaceScopedSchema, ChatSessionUpdate
):
    """Stored chat session."""

    suggest_title: bool = Field(
        True,
        description=(
            "When title is unset, use the chat_session_title prompt after each turn "
            "until the model reports the topic is specific enough"
        ),
    )


class ChatThreadCreate(BaseModel):
    """Create another thread inside a session (different model / branch)."""

    title: str | None = Field(None, description="Optional thread title")
    chat_model: str | None = Field(
        None,
        description="OpenRouter model id; falls back to server default when omitted",
    )


class ChatThreadSchema(UserOwnedEntitySchema, WorkspaceScopedSchema, ChatThreadCreate):
    """Stored chat thread."""

    session_uid: str = Field(..., description="Owning session uid")
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
        True,
        description=(
            "If true, call OpenRouter with full thread history "
            "and append assistant message"
        ),
    )
    stream: bool = Field(
        False,
        description="When generate_reply is true, stream SSE chunks like OpenRouter",
    )


class ChatMessageSchema(UserOwnedEntitySchema, WorkspaceScopedSchema):
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


class ChatQuickStartCreate(ChatMessageCreate):
    """Send the first message; creates a session and thread automatically."""

    title: str | None = Field(
        None,
        description="Optional session title; omit to auto-suggest from the message",
    )
    suggest_title: bool = Field(
        True,
        description=(
            "When title is omitted, use the chat_session_title prompt to decide "
            "if the conversation is specific enough for a title"
        ),
    )
    chat_model: str | None = Field(
        None,
        description="OpenRouter model id for the new thread",
    )
    thread_title: str | None = Field(
        None,
        description="Title for the first thread (default: Thread 1)",
    )


class ChatQuickStartResponse(BaseModel):
    """Reply to POST /chat/messages (quick-start)."""

    session: ChatSessionSchema
    thread: ChatThreadSchema
    user_message: ChatMessageSchema
    assistant_message: ChatMessageSchema | None = None


class ChatQuickThreadCreate(ChatMessageCreate):
    """Send the first message on a new thread inside an existing session."""

    thread_title: str | None = Field(
        None,
        description="Optional thread title; omit to auto-suggest from the message",
    )
    suggest_thread_title: bool = Field(
        True,
        description="When thread_title is omitted, ask the model for a short title",
    )
    chat_model: str | None = Field(
        None,
        description="OpenRouter model id for the new thread",
    )


class ChatQuickThreadResponse(BaseModel):
    """Reply to POST /chat/sessions/{session_uid}/messages (quick new thread)."""

    thread: ChatThreadSchema
    user_message: ChatMessageSchema
    assistant_message: ChatMessageSchema | None = None
