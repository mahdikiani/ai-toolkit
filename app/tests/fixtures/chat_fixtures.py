"""Fixtures for chat sessions, threads, and messages."""

import contextlib
from collections.abc import AsyncGenerator

import pytest_asyncio

from apps.language.chat.models import ChatMessage, ChatSession, ChatThread


@pytest_asyncio.fixture
async def chat_session(mock_user: dict[str, str]) -> AsyncGenerator[ChatSession]:
    """Create a test chat session."""
    session = await ChatSession.create_item({
        "user_id": mock_user["user_id"],
        "title": "Test Session",
    })
    yield session
    with contextlib.suppress(Exception):
        await session.delete()


@pytest_asyncio.fixture
async def chat_thread(
    chat_session: ChatSession, mock_user: dict[str, str]
) -> AsyncGenerator[ChatThread]:
    """Create a test chat thread within a session."""
    thread = await ChatThread.create_item({
        "session_uid": chat_session.uid,
        "user_id": mock_user["user_id"],
        "title": "Test Thread",
        "chat_model": "openai/gpt-4o-mini",
    })
    yield thread
    with contextlib.suppress(Exception):
        await thread.delete()


@pytest_asyncio.fixture
async def chat_messages(
    chat_thread: ChatThread, mock_user: dict[str, str]
) -> AsyncGenerator[list[ChatMessage]]:
    """Create test chat messages in a thread."""
    messages = []
    for role, content in [
        ("user", "Hello, how are you?"),
        ("assistant", "I'm doing well, thank you!"),
        ("user", "Can you help me?"),
    ]:
        msg = await ChatMessage.create_item({
            "thread_uid": chat_thread.uid,
            "user_id": mock_user["user_id"],
            "role": role,
            "content": content,
        })
        messages.append(msg)

    yield messages

    for msg in messages:
        with contextlib.suppress(Exception):
            await msg.delete()
