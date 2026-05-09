"""OpenRouter proxy and chat completion helpers."""

from collections.abc import AsyncIterator

from fastapi import HTTPException

from server.config import Settings
from utils import openrouter as openrouter_client

from .models import ChatMessage, ChatThread


def openrouter_headers() -> dict[str, str]:
    """Return Authorization and content headers for OpenRouter."""
    try:
        return openrouter_client.build_headers()
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured",
        ) from e


async def proxy_chat_completions(
    payload: dict,
) -> tuple[bytes, str | None, int]:
    """Forward a chat completion request to OpenRouter (non-streaming)."""
    try:
        resp = await openrouter_client.post_chat_completion_unchecked(payload)
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured",
        ) from None
    ctype = resp.headers.get("content-type")
    return resp.content, ctype, resp.status_code


async def proxy_chat_completions_raw_stream(payload: dict) -> AsyncIterator[bytes]:
    """Forward streaming chat completions; yields raw SSE bytes from OpenRouter."""
    try:
        async for chunk in openrouter_client.stream_chat_completion_bytes(payload):
            yield chunk
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured",
        ) from None
    except openrouter_client.OpenRouterError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e


async def iter_openrouter_sse_deltas(payload: dict) -> AsyncIterator[str]:
    """Yield text deltas from an OpenRouter streaming chat completion."""
    try:
        async for delta in openrouter_client.stream_chat_deltas(payload):
            yield delta
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured",
        ) from None
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


def thread_model(thread: ChatThread) -> str:
    """Return the model ID for a thread, falling back to server default."""
    return thread.chat_model or Settings.default_model


async def messages_as_openrouter(thread: ChatThread) -> list[dict[str, str]]:
    """Fetch thread messages formatted for OpenRouter API."""
    rows = await ChatMessage.list_items(
        tenant_id=thread.tenant_id,
        user_id=thread.user_id,
        thread_uid=thread.uid,
        offset=0,
        limit=500,
        sort_field="created_at",
        sort_direction=1,
        is_deleted=False,
    )
    return [{"role": m.role, "content": m.content} for m in rows]


async def complete_assistant_message(
    *,
    thread: ChatThread,
    user_id: str,
    tenant_id: str,
) -> ChatMessage:
    """Call OpenRouter (non-stream), persist assistant reply."""
    msgs = await messages_as_openrouter(thread)
    if not msgs:
        raise HTTPException(status_code=400, detail="Thread has no messages yet")

    payload = {
        "model": thread_model(thread),
        "messages": msgs,
        "temperature": 0.7,
    }

    try:
        raw_json = await openrouter_client.complete_chat_json(payload)
    except ValueError:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY is not configured",
        ) from None
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    choices = raw_json.get("choices") or []
    content = ""
    if choices:
        content = (choices[0].get("message") or {}).get("content") or ""

    return await ChatMessage.create_item({
        "thread_uid": thread.uid,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": "assistant",
        "content": content.strip(),
        "completion_extra": {
            "model": raw_json.get("model"),
            "usage": raw_json.get("usage"),
        },
    })
