"""Shared OpenAI-compatible chat completion proxy + metering helpers."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi_mongo_base.core.exceptions import BaseHTTPException

from server.config import Settings
from utils.billing import finance
from utils.integrations.openrouter import (
    OpenRouterError,
    extract_provider_meta,
    post_chat_completion_unchecked,
    stream_chat_completion_bytes,
)

logger = logging.getLogger(__name__)


def openai_error(status: int, code: str, message: str) -> BaseHTTPException:
    """Build an OpenAI-shaped BaseHTTPException."""
    return BaseHTTPException(
        status_code=status,
        error_code=code,
        detail=message,
        message={"en": message},
    )


def openai_chat_id() -> str:
    """Return a chatcmpl-* response id."""
    return f"chatcmpl-{uuid4().hex[:12]}"


def estimate_chat_cost(body: dict) -> float:
    """Rough token estimate for quota pre-check."""
    total_chars = 0
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(part.get("text", ""))
    estimated_tokens = max(100, total_chars // 4)
    pricing = finance.pricing_config()["text"]
    per_1k = float(pricing.get("default_per_1k_tokens", 1.0))
    markup = float(pricing.get("markup", 1.0))
    model_cfg = pricing.get("models", {}).get(body.get("model", ""), {})
    if isinstance(model_cfg, dict):
        per_1k = float(model_cfg.get("per_1k_tokens", per_1k))
    return (estimated_tokens / 1000) * per_1k * markup


async def meter_chat_usage(
    *,
    user_id: str,
    model: str,
    usage: dict | None = None,
    provider_meta: dict | None = None,
    content: str | None = None,
    service: str = "openai_compat",
) -> None:
    """Meter chat usage from provider meta, usage dict, or content estimate."""
    meta = provider_meta or {}
    if usage and "usage" not in meta:
        meta = {**meta, "usage": usage}
    if not meta.get("model"):
        meta = {**meta, "model": model}

    amount = finance.estimate_text_cost(
        model=str(meta.get("model") or model),
        usage=meta.get("usage") if isinstance(meta.get("usage"), dict) else usage,
        raw_cost=meta.get("raw_cost"),
    )
    if amount <= 0 and content:
        # Fallback estimate from streamed text when provider omitted usage.
        approx_tokens = max(50, len(content) // 4)
        pricing = finance.pricing_config()["text"]
        per_1k = float(pricing.get("default_per_1k_tokens", 1.0))
        markup = float(pricing.get("markup", 1.0))
        amount = (approx_tokens / 1000) * per_1k * markup

    await finance.meter_cost(
        user_id,
        amount,
        meta_data={"service": service, "provider_meta": meta},
    )


async def handle_non_stream_chat(
    body: dict,
    *,
    user_id: str,
    model: str,
    service: str = "openai_compat",
) -> JSONResponse:
    """Non-streaming completion with quota check + metering."""
    estimated = estimate_chat_cost(body)
    await finance.check_quota_or_error(user_id, estimated)

    resp = await post_chat_completion_unchecked(body)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise openai_error(resp.status_code, "upstream_error", str(detail))

    data = resp.json()
    choice = data.get("choices", [{}])[0]
    usage = data.get("usage", {})
    resp_id = openai_chat_id()

    try:
        provider_meta = extract_provider_meta(data, provider="openrouter")
        await meter_chat_usage(
            user_id=user_id,
            model=str(provider_meta.get("model", model)),
            usage=usage if isinstance(usage, dict) else None,
            provider_meta=provider_meta,
            service=service,
        )
    except Exception:
        logger.exception("Failed to meter openai_compat non-stream usage")

    return JSONResponse(
        content={
            "id": resp_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", model),
            "choices": [
                {
                    "index": choice.get("index", 0),
                    "message": {
                        "role": "assistant",
                        "content": choice.get("message", {}).get("content", ""),
                    },
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
    )


def _sse_data_payload(line: str) -> str | None:
    """Normalize one SSE line to a JSON payload string, or None to skip."""
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if line.startswith("data: "):
        line = line[6:]
    return line


def _openai_content_chunk(
    *,
    resp_id: str,
    created: int,
    model: str,
    chunk_data: dict,
) -> bytes | None:
    """Build an OpenAI SSE chunk from an upstream delta payload."""
    choices = chunk_data.get("choices", [])
    if not choices:
        return None
    delta = choices[0].get("delta", {}) or {}
    content = delta.get("content", "")
    openai_chunk = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": chunk_data.get("model", model),
        "choices": [
            {
                "index": choices[0].get("index", 0),
                "delta": {"content": content} if content else {},
                "finish_reason": choices[0].get("finish_reason"),
            }
        ],
    }
    return f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n".encode()


def _update_usage_from_chunk(
    chunk_data: dict,
    full_content: list[str],
) -> tuple[dict | None, dict | None]:
    """Collect usage/meta and append streamed content deltas."""
    usage = None
    meta = None
    if isinstance(chunk_data.get("usage"), dict):
        usage = chunk_data["usage"]
        meta = extract_provider_meta(chunk_data, provider="openrouter")
    choices = chunk_data.get("choices", [])
    if choices:
        content = (choices[0].get("delta", {}) or {}).get("content", "")
        if content:
            full_content.append(content)
    return usage, meta


async def _iter_openai_sse_bytes(
    body: dict,
    *,
    resp_id: str,
    created: int,
    model: str,
    full_content: list[str],
) -> AsyncIterator[tuple[bytes, dict | None, dict | None]]:
    """Yield OpenAI SSE payloads plus latest usage/meta."""
    last_usage: dict | None = None
    last_meta: dict | None = None
    async for raw_bytes in stream_chat_completion_bytes(body):
        for raw_line in raw_bytes.decode(errors="replace").split("\n"):
            line = _sse_data_payload(raw_line)
            if line is None:
                continue
            if line == "[DONE]":
                return
            try:
                chunk_data = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage, meta = _update_usage_from_chunk(chunk_data, full_content)
            if usage is not None:
                last_usage, last_meta = usage, meta
            out = _openai_content_chunk(
                resp_id=resp_id,
                created=created,
                model=model,
                chunk_data=chunk_data,
            )
            if out is not None:
                yield out, last_usage, last_meta


async def _stream_chat_events(
    body: dict,
    *,
    user_id: str,
    service: str,
) -> AsyncIterator[bytes]:
    """Yield OpenAI SSE bytes and meter usage when the stream ends."""
    resp_id = openai_chat_id()
    created = int(time.time())
    model = body.get("model", Settings.default_model)
    full_content: list[str] = []
    last_usage: dict | None = None
    last_provider_meta: dict | None = None

    role_chunk = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
        ],
    }
    yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n".encode()

    try:
        async for out, usage, meta in _iter_openai_sse_bytes(
            body,
            resp_id=resp_id,
            created=created,
            model=model,
            full_content=full_content,
        ):
            last_usage, last_provider_meta = usage, meta
            yield out
    except OpenRouterError as e:
        err = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        }
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode()
        yield (
            "data: "
            + json.dumps(
                {"error": {"message": str(e), "type": "upstream_error"}},
                ensure_ascii=False,
            )
            + "\n\n"
        ).encode()
    finally:
        yield b"data: [DONE]\n\n"
        try:
            await meter_chat_usage(
                user_id=user_id,
                model=model,
                usage=last_usage,
                provider_meta=last_provider_meta,
                content="".join(full_content),
                service=service,
            )
        except Exception:
            logger.exception("Failed to meter openai_compat stream usage")


async def handle_stream_chat(
    body: dict,
    *,
    user_id: str,
    service: str = "openai_compat",
) -> StreamingResponse:
    """Streaming completion with quota pre-check and post-stream metering."""
    estimated = estimate_chat_cost(body)
    await finance.check_quota_or_error(user_id, estimated)
    return StreamingResponse(
        _stream_chat_events(body, user_id=user_id, service=service),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
