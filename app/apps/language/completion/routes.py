"""
OpenAI-compatible completion proxy routes.

Prefer the canonical mounted surface at ``/api/ai/v1/openai/v1/*``.
This module remains for shared helpers / optional alias mounting.
"""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse
from fastapi_mongo_base.errors import BadRequestError
from usso import UserData

from utils.usso import get_usso

from .services import proxy_chat_completions, proxy_chat_completions_raw_stream

router = APIRouter(prefix="/chat", tags=["Completion"])
auth_dependency = get_usso(raise_exception=True)


@router.post("/completions")
async def openai_compatible_chat_completions(
    request: Request,
    user: UserData = Depends(auth_dependency),
) -> Response:
    """Proxy an OpenAI-compatible chat completion (alias of openai_compat)."""
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise BadRequestError(
            error_code="invalid_json",
            detail="Invalid JSON body",
            message={"en": "Invalid JSON body"},
        ) from e

    if not isinstance(body, dict):
        raise BadRequestError(
            error_code="invalid_body",
            detail="Body must be a JSON object",
            message={"en": "Body must be a JSON object"},
        )

    user_id = getattr(user, "user_id", None) or getattr(user, "uid", "")

    if body.get("stream") is True:

        async def passthrough() -> AsyncIterator[bytes]:
            async for chunk in proxy_chat_completions_raw_stream(body, user_id=user_id):
                yield chunk

        return StreamingResponse(passthrough(), media_type="text/event-stream")

    raw, ctype, status = await proxy_chat_completions(body, user_id=user_id)
    return Response(
        content=raw,
        status_code=status,
        media_type=ctype or "application/json",
    )
