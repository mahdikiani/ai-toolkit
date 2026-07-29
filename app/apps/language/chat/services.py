"""OpenRouter proxy and chat completion helpers."""

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from apps.language.promptic.engine import PromptEngine, load_data
from apps.language.promptic.services import call_openrouter
from apps.language.shared.exceptions import (
    OpenRouterHttpError,
    OpenRouterNotConfiguredError,
    OpenRouterUpstreamError,
    ThreadHasNoMessagesError,
)
from server.config import Settings
from utils.billing import finance
from utils.integrations import openrouter as openrouter_client

from .models import ChatMessage, ChatSession, ChatThread

logger = logging.getLogger(__name__)

SESSION_TITLE_PROMPT = "chat_session_title.yaml"

# After this many messages with still no LLM-judged title, guarantee a
# fallback title (see maybe_apply_session_title_if_ready) instead of
# leaving the session untitled indefinitely.
_TITLE_FALLBACK_AFTER_MESSAGES = 4


@dataclass(frozen=True)
class SessionTitleSuggestion:
    """Result of the session-title prompt."""

    has_title: bool
    title: str | None = None


async def bootstrap_session(
    *,
    user_id: str,
    title: str | None = None,
    thread_title: str | None = None,
    chat_model: str | None = None,
    suggest_title: bool = True,
) -> tuple[ChatSession, ChatThread]:
    """Create a session and its first thread."""
    session = await ChatSession.create_item({
        "title": title,
        "suggest_title": suggest_title,
        "user_id": user_id,
    })
    thread = await ChatThread.create_item({
        "session_uid": session.uid,
        "title": thread_title or "Thread 1",
        "chat_model": chat_model,
        "user_id": user_id,
    })
    session.active_thread_uid = thread.uid
    await session.save()
    return session, thread


async def evaluate_session_title(
    *,
    user_id: str,
    thread: ChatThread,
) -> SessionTitleSuggestion:
    """Use promptic to decide if the conversation warrants a session title."""
    messages = await messages_as_openrouter(thread)
    if not messages:
        return SessionTitleSuggestion(has_title=False)

    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / SESSION_TITLE_PROMPT
    if not prompt_path.exists():
        return SessionTitleSuggestion(has_title=False)

    conversation = "\n".join(
        f"{message['role'].capitalize()}: {message['content']}" for message in messages
    )

    try:
        meta = load_data(prompt_path)
        if not isinstance(meta, dict):
            return SessionTitleSuggestion(has_title=False)

        engine = PromptEngine(base_dir=prompts_dir)
        system_prompt, user_prompt, response_format = engine.generate(
            prompt_path,
            {"conversation": conversation},
        )

        model = meta.get("model") or Settings.title_model
        temperature = float(meta.get("temperature", 0.3))
        max_tokens = meta.get("max_tokens")
        if max_tokens is not None:
            max_tokens = int(max_tokens)

        openrouter_result = await call_openrouter(
            system_prompt,
            user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            return_meta=True,
        )
        if isinstance(openrouter_result, tuple):
            raw_content, provider_meta = openrouter_result
        else:
            raw_content = openrouter_result
            provider_meta = {}

        parsed = json.loads(raw_content)
        block = parsed.get("session_title", parsed)
        has_title = bool(block.get("has_title"))
        title = (block.get("title") or "").strip().strip("\"'") or None

        provider_usage = provider_meta.get("usage")
        raw_cost = provider_meta.get("raw_cost")
        amount = finance.estimate_text_cost(
            model=str(provider_meta.get("model") or model),
            usage=provider_usage if isinstance(provider_usage, dict) else None,
            raw_cost=_text_cost_value(raw_cost),
        )
        try:
            await finance.meter_cost(
                user_id,
                amount,
                meta_data={
                    "service": "chat",
                    "kind": "session_title",
                    "prompt": SESSION_TITLE_PROMPT,
                    "provider_meta": provider_meta,
                },
            )
        except Exception:
            logger.exception("Failed to meter session-title usage for %s", user_id)

        if not has_title or not title:
            return SessionTitleSuggestion(has_title=False)
        return SessionTitleSuggestion(has_title=True, title=title[:120])
    except (ValueError, RuntimeError, json.JSONDecodeError, TypeError, KeyError):
        return SessionTitleSuggestion(has_title=False)


async def maybe_apply_session_title_if_ready(
    *,
    session: ChatSession,
    thread: ChatThread,
    user_id: str,
) -> ChatSession:
    """
    Set session.title once the topic is clear -- guaranteed eventually.

    evaluate_session_title asks an LLM whether the conversation is
    specific enough yet, and is re-run on every message while
    session.title stays None -- by design, it keeps saying no for
    generic openers ("hi", "سلام"). Left alone, a short or ambiguous
    conversation could stay "بدون عنوان" (untitled) indefinitely. After
    a couple of exchanges without a natural title, fall back to the
    first user message itself so every session ends up with a real
    title, not just the ones the LLM judges "interesting enough."
    """
    if session.title is not None or not session.suggest_title:
        return session

    suggestion = await evaluate_session_title(user_id=user_id, thread=thread)
    if suggestion.has_title and suggestion.title:
        session.title = suggestion.title
        await session.save()
        return session

    messages = await messages_as_openrouter(thread)
    if len(messages) >= _TITLE_FALLBACK_AFTER_MESSAGES:
        first_user_message = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        fallback_title = first_user_message.strip()[:60]
        if fallback_title:
            session.title = fallback_title
            await session.save()

    return session


async def suggest_title_from_exchange(
    *,
    user_id: str,
    user_content: str,
    assistant_content: str | None = None,
    model: str | None = None,
    kind: Literal["session", "thread"] = "session",
) -> str | None:
    """Ask OpenRouter for a short title based on a single exchange."""
    model = model or Settings.title_model
    exchange = f"User: {user_content.strip()}"
    if assistant_content:
        exchange += f"\nAssistant: {assistant_content.strip()}"

    subject = "chat session" if kind == "session" else "chat thread"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Suggest a very short {subject} title (max 6 words, no quotes). "
                    "Reply with only the title."
                ),
            },
            {"role": "user", "content": exchange},
        ],
        "temperature": 0.3,
        "max_tokens": 30,
    }

    try:
        raw_json = await openrouter_client.complete_chat_json(payload)
    except (ValueError, RuntimeError):
        return None

    choices = raw_json.get("choices") or []
    title = ""
    if choices:
        title = ((choices[0].get("message") or {}).get("content") or "").strip()
    title = title.strip("\"'").strip()
    if not title:
        return None

    provider_meta = openrouter_client.extract_provider_meta(
        raw_json,
        provider="openrouter",
    )
    provider_usage = provider_meta.get("usage")
    raw_cost = provider_meta.get("raw_cost")
    amount = finance.estimate_text_cost(
        model=str(provider_meta.get("model") or model),
        usage=provider_usage if isinstance(provider_usage, dict) else None,
        raw_cost=_text_cost_value(raw_cost),
    )
    try:
        await finance.meter_cost(
            user_id,
            amount,
            meta_data={
                "service": "chat",
                "kind": f"{kind}_title",
                "provider_meta": provider_meta,
            },
        )
    except Exception:
        logger.exception("Failed to meter %s-title usage for %s", kind, user_id)
    return title[:120]


async def maybe_apply_suggested_thread_title(
    *,
    thread: ChatThread,
    user_id: str,
    user_content: str,
    assistant_content: str | None,
    title: str | None,
    suggest_title: bool,
    model: str | None = None,
) -> ChatThread:
    """Set thread.title from LLM when no explicit title was provided."""
    if title is not None:
        return thread
    if not suggest_title:
        return thread

    suggested = await suggest_title_from_exchange(
        user_id=user_id,
        user_content=user_content,
        assistant_content=assistant_content,
        model=model or Settings.title_model,
        kind="thread",
    )
    if suggested:
        thread.title = suggested
        await thread.save()
    return thread


def openrouter_headers() -> dict[str, str]:
    """Return Authorization and content headers for OpenRouter."""
    try:
        return openrouter_client.build_headers()
    except ValueError as e:
        raise OpenRouterNotConfiguredError() from e


async def proxy_chat_completions(
    payload: dict,
) -> tuple[bytes, str | None, int]:
    """Forward a chat completion request to OpenRouter (non-streaming)."""
    try:
        resp = await openrouter_client.post_chat_completion_unchecked(payload)
    except ValueError:
        raise OpenRouterNotConfiguredError() from None
    ctype = resp.headers.get("content-type")
    return resp.content, ctype, resp.status_code


async def proxy_chat_completions_raw_stream(payload: dict) -> AsyncIterator[bytes]:
    """Forward streaming chat completions; yields raw SSE bytes from OpenRouter."""
    try:
        async for chunk in openrouter_client.stream_chat_completion_bytes(payload):
            yield chunk
    except ValueError:
        raise OpenRouterNotConfiguredError() from None
    except openrouter_client.OpenRouterError as e:
        raise OpenRouterHttpError(e.status_code, e.detail) from e


async def iter_openrouter_sse_deltas(payload: dict) -> AsyncIterator[str]:
    """Yield text deltas from an OpenRouter streaming chat completion."""
    try:
        async for delta in openrouter_client.stream_chat_deltas(payload):
            yield delta
    except ValueError:
        raise OpenRouterNotConfiguredError() from None
    except RuntimeError as e:
        raise OpenRouterUpstreamError(str(e)) from e


def thread_model(thread: ChatThread) -> str:
    """Return the model ID for a thread, falling back to server default."""
    return thread.chat_model or Settings.default_model


async def messages_as_openrouter(thread: ChatThread) -> list[dict[str, str]]:
    """Fetch thread messages formatted for OpenRouter API."""
    rows = await ChatMessage.list_items(
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
) -> ChatMessage:
    """Call OpenRouter (non-stream), persist assistant reply."""
    msgs = await messages_as_openrouter(thread)
    if not msgs:
        raise ThreadHasNoMessagesError()

    model = thread_model(thread)
    payload = {
        "model": model,
        "messages": msgs,
        "temperature": 0.7,
    }

    estimated = finance.estimate_text_cost(
        model=model,
        usage={"total_tokens": max(100, sum(len(m["content"]) for m in msgs) // 4)},
    )
    await finance.check_quota(user_id, estimated, raise_exception=True)

    try:
        raw_json = await openrouter_client.complete_chat_json(payload)
    except ValueError:
        raise OpenRouterNotConfiguredError() from None
    except RuntimeError as e:
        raise OpenRouterUpstreamError(str(e)) from e

    choices = raw_json.get("choices") or []
    content = ""
    if choices:
        content = (choices[0].get("message") or {}).get("content") or ""
    provider_meta = openrouter_client.extract_provider_meta(
        raw_json,
        provider="openrouter",
    )
    provider_usage = provider_meta.get("usage")
    raw_cost = provider_meta.get("raw_cost")
    amount = finance.estimate_text_cost(
        model=str(provider_meta.get("model") or payload["model"]),
        usage=provider_usage if isinstance(provider_usage, dict) else None,
        raw_cost=_text_cost_value(raw_cost),
    )
    try:
        usage = await finance.meter_cost(
            user_id,
            amount,
            meta_data={
                "service": "chat",
                "thread_uid": thread.uid,
                "provider_meta": provider_meta,
            },
        )
    except Exception:
        logger.exception("Failed to meter chat usage for thread %s", thread.uid)
        usage = None

    return await ChatMessage.create_item({
        "thread_uid": thread.uid,
        "user_id": user_id,
        "role": "assistant",
        "content": content.strip(),
        "completion_extra": {
            "provider_meta": provider_meta,
            "usage_amount": float(usage.amount) if usage else amount,
            "usage_id": usage.uid if usage else None,
        },
    })


AfterReplyHook = Callable[[], Awaitable[None]]


def _text_cost_value(value: object) -> int | float | str | None:
    """Return a provider cost value accepted by the finance service."""
    if isinstance(value, (int, float, str)):
        return value
    return None
