"""OpenAI-compatible image generation/edits proxied through OpenRouter."""

from __future__ import annotations

import logging

import httpx
from fastapi.responses import JSONResponse

from server.config import Settings
from utils.billing import finance
from utils.integrations.openrouter import build_headers, resolve_api_key

from .services import openai_error

logger = logging.getLogger(__name__)


def _estimate_image_cost(body: dict) -> float:
    return finance.estimate_image_cost(
        count=max(1, int(body.get("n", 1))),
        model=body.get("model"),
    )


async def create_generation(
    body: dict, *, user_id: str, workspace_id: str | None = None
) -> JSONResponse:
    """Generate images via OpenRouter's images/generations endpoint."""
    prompt = body.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise openai_error(400, "invalid_request_error", "prompt is required")

    model = body.get("model", "openai/dall-e-3")
    n = body.get("n", 1)
    size = body.get("size", "1024x1024")
    quality = body.get("quality", "standard")
    response_format = body.get("response_format", "url")

    estimated = _estimate_image_cost(body)
    await finance.check_quota_or_error(user_id, estimated, workspace_id=workspace_id)

    try:
        resolve_api_key()
    except Exception as exc:
        raise openai_error(503, "service_unavailable", str(exc)) from exc

    payload = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "quality": quality,
        "response_format": response_format,
    }
    url = f"{Settings.openrouter_base_url.rstrip('/')}/images/generations"
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload, headers=build_headers())

    if resp.status_code >= 400:
        raise openai_error(resp.status_code, "upstream_error", resp.text[:500])

    data = resp.json()

    try:
        await finance.meter_cost(
            user_id,
            estimated,
            meta_data={
                "service": "openai_compat_image_generation",
                "model": model,
                "n": n,
            },
            workspace_id=workspace_id,
        )
    except Exception:
        logger.exception("Failed to meter image generation usage")

    return JSONResponse(content=data)


async def create_edit(
    body: dict,
    *,
    user_id: str,
    image_bytes: bytes | None = None,
    mask_bytes: bytes | None = None,
    workspace_id: str | None = None,
) -> JSONResponse:
    """Edit an image via OpenRouter's images/edits endpoint."""
    prompt = body.get("prompt")
    if not prompt or not isinstance(prompt, str):
        raise openai_error(400, "invalid_request_error", "prompt is required")

    model = body.get("model", "openai/dall-e-2")
    n = body.get("n", 1)
    size = body.get("size", "1024x1024")
    response_format = body.get("response_format", "url")

    estimated = _estimate_image_cost(body)
    await finance.check_quota_or_error(user_id, estimated, workspace_id=workspace_id)

    try:
        resolve_api_key()
    except Exception as exc:
        raise openai_error(503, "service_unavailable", str(exc)) from exc

    url = f"{Settings.openrouter_base_url.rstrip('/')}/images/edits"

    data = {
        "model": model,
        "prompt": prompt,
        "n": str(n),
        "size": size,
        "response_format": response_format,
    }

    files: dict = {"prompt": (None, prompt)}
    if image_bytes:
        files["image"] = ("image.png", image_bytes, "image/png")
    if mask_bytes:
        files["mask"] = ("mask.png", mask_bytes, "image/png")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, data=data, files=files, headers=build_headers())

    if resp.status_code >= 400:
        raise openai_error(resp.status_code, "upstream_error", resp.text[:500])

    data = resp.json()

    try:
        await finance.meter_cost(
            user_id,
            estimated,
            meta_data={
                "service": "openai_compat_image_edit",
                "model": model,
                "n": n,
            },
            workspace_id=workspace_id,
        )
    except Exception:
        logger.exception("Failed to meter image edit usage")

    return JSONResponse(content=data)
