"""OpenAI-compatible REST API: /v1/models, /v1/chat/completions."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi_mongo_base.core.exceptions import BaseHTTPException
from usso import UserData

from server.config import Settings
from utils.billing import finance
from utils.integrations.openrouter import (
    OpenRouterError,
    chat_completions_url,
    post_chat_completion_unchecked,
    stream_chat_completion_bytes,
)
from utils.usso import get_usso

router = APIRouter(prefix="/openai/v1", tags=["OpenAI Compatible"])
auth = get_usso(raise_exception=True)

AVAILABLE_MODELS = [
    {
        "id": Settings.default_model,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "openrouter",
    },
]

MODELS_CONFIG_RAW = (getattr(Settings, "openai_compat_models", "") or "").strip()
if MODELS_CONFIG_RAW:
    extra_ids = [m.strip() for m in MODELS_CONFIG_RAW.split(",") if m.strip()]
    existing = {m["id"] for m in AVAILABLE_MODELS}
    for mid in extra_ids:
        if mid not in existing:
            existing.add(mid)
            AVAILABLE_MODELS.append({
                "id": mid,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openrouter",
            })


def _openai_error(status: int, code: str, message: str) -> BaseHTTPException:
    return BaseHTTPException(
        status_code=status,
        error_code=code,
        detail=message,
        message={"en": message},
    )


def _openai_id() -> str:
    return f"chatcmpl-{uuid4().hex[:12]}"


@router.get("/models")
async def list_models(user: UserData = Depends(auth)) -> dict:
    """Return available models in OpenAI format."""
    return {
        "object": "list",
        "data": AVAILABLE_MODELS,
    }


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    user: UserData = Depends(auth),
):
    """OpenAI-compatible chat completions endpoint.

    Accepts the standard OpenAI /v1/chat/completions request body.
    Proxies to OpenRouter, meters cost, and returns OpenAI-format response.
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise _openai_error(400, "invalid_request_error", "Invalid JSON body")

    if not isinstance(body, dict):
        raise _openai_error(400, "invalid_request_error", "Body must be a JSON object")

    model = body.get("model", Settings.default_model)
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens")
    top_p = body.get("top_p", 1.0)

    if not messages:
        raise _openai_error(400, "invalid_request_error", "messages is required")

    # Convert to OpenRouter format
    or_body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        or_body["max_tokens"] = max_tokens

    user_id = getattr(user, "uid", None) or getattr(user, "user_id", "")

    if stream:
        return await _handle_stream(or_body, user_id)

    return await _handle_non_stream(or_body, user_id, model)


async def _handle_non_stream(
    body: dict,
    user_id: str,
    model: str,
) -> JSONResponse:
    """Non-streaming completion: returns standard OpenAI JSON response."""
    # check quota
    estimated = _estimate_cost(body)
    await finance.check_quota(user_id, estimated, raise_exception=True)

    resp = await post_chat_completion_unchecked(body)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise _openai_error(
            resp.status_code,
            "upstream_error",
            str(detail),
        )

    data = resp.json()
    choice = data.get("choices", [{}])[0]
    usage = data.get("usage", {})

    resp_id = _openai_id()

    # meter cost
    try:
        from utils.integrations.openrouter import extract_provider_meta

        provider_meta = extract_provider_meta(data, provider="openrouter")
        amount = finance.estimate_text_cost(
            model=str(provider_meta.get("model", model)),
            usage=provider_meta.get("usage"),
            raw_cost=provider_meta.get("raw_cost"),
        )
        await finance.meter_cost(
            user_id,
            amount,
            meta_data={
                "service": "openai_compat",
                "provider_meta": provider_meta,
            },
        )
    except Exception:
        pass

    return JSONResponse(content={
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
    })


async def _handle_stream(
    body: dict,
    user_id: str,
) -> StreamingResponse:
    """Streaming completion: yields OpenAI-format SSE chunks."""

    async def event_stream() -> AsyncIterator[bytes]:
        resp_id = _openai_id()
        created = int(time.time())
        model = body.get("model", Settings.default_model)
        full_content: list[str] = []

        # yield role chunk
        role_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n".encode()

        try:
            async for raw_bytes in stream_chat_completion_bytes(body):
                decoded = raw_bytes.decode(errors="replace")
                for line in decoded.split("\n"):
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract delta
                    choices = chunk_data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {}) or {}
                        content = delta.get("content", "")
                        if content:
                            full_content.append(content)

                        # Rewrite into OpenAI format
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
                        yield f"data: {json.dumps(openai_chunk, ensure_ascii=False)}\n\n".encode()

        except OpenRouterError as e:
            error_chunk = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "error",
                    }
                ],
            }
            yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n".encode()
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'upstream_error'}}, ensure_ascii=False)}\n\n".encode()
        finally:
            yield b"data: [DONE]\n\n"

        # Meter cost after streaming completes
        try:
            full = "".join(full_content)
        except Exception:
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _estimate_cost(body: dict) -> float:
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
    per_1k = float(
        pricing.get("default_per_1k_tokens", 1.0)
    )
    markup = float(pricing.get("markup", 1.0))
    model_cfg = pricing.get("models", {}).get(body.get("model", ""), {})
    if isinstance(model_cfg, dict):
        per_1k = float(model_cfg.get("per_1k_tokens", per_1k))
    return (estimated_tokens / 1000) * per_1k * markup
