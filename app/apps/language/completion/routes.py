"""OpenAI-compatible completion proxy routes."""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from usso.integrations.fastapi import USSOAuthentication

from .services import proxy_chat_completions, proxy_chat_completions_raw_stream

router = APIRouter(prefix="/completion", tags=["Completion"])
auth_dependency = USSOAuthentication()


async def _require_user(request: Request) -> object:
    """Require USSO authentication and return the authenticated user."""
    return await auth_dependency(request)


@router.post("/v1/chat/completions")
async def openai_compatible_chat_completions(request: Request):  # noqa: ANN201
    """Proxy an OpenAI-compatible chat completion request to configured providers."""
    user = await _require_user(request)
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from e

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    if body.get("stream") is True:

        async def passthrough() -> AsyncIterator[bytes]:
            async for chunk in proxy_chat_completions_raw_stream(body):
                yield chunk

        return StreamingResponse(passthrough(), media_type="text/event-stream")

    raw, ctype, status = await proxy_chat_completions(
        body,
        user_id=getattr(user, "user_id", None) or getattr(user, "uid", ""),
    )
    return Response(
        content=raw,
        status_code=status,
        media_type=ctype or "application/json",
    )
