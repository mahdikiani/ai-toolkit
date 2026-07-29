"""OpenAI-compatible REST API under /openai/v1."""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import Response
from usso import UserData

from server.config import Settings
from utils.usso import get_usso

from . import audio as audio_api
from . import images as images_api
from . import services

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


@router.get("/models")
async def list_models(user: UserData = Depends(auth)) -> dict:
    """Return available models in OpenAI format."""
    return {"object": "list", "data": AVAILABLE_MODELS}


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    user: UserData = Depends(auth),
) -> object:
    """Proxy OpenAI-compatible chat completions with metering."""
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise services.openai_error(
            400, "invalid_request_error", "Invalid JSON body"
        ) from exc

    if not isinstance(body, dict):
        raise services.openai_error(
            400, "invalid_request_error", "Body must be a JSON object"
        )

    model = body.get("model", Settings.default_model)
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens")
    top_p = body.get("top_p", 1.0)

    if not messages:
        raise services.openai_error(
            400, "invalid_request_error", "messages is required"
        )

    or_body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        or_body["max_tokens"] = max_tokens
    if stream:
        or_body["stream"] = True

    user_id = getattr(user, "uid", None) or getattr(user, "user_id", "")

    if stream:
        return await services.handle_stream_chat(or_body, user_id=user_id)

    return await services.handle_non_stream_chat(or_body, user_id=user_id, model=model)


@router.post("/audio/speech")
async def audio_speech(
    request: Request,
    user: UserData = Depends(auth),
) -> Response:
    """OpenAI-compatible TTS: POST /audio/speech → OpenRouter audio/speech."""
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise services.openai_error(
            400, "invalid_request_error", "Invalid JSON body"
        ) from exc
    if not isinstance(body, dict):
        raise services.openai_error(
            400, "invalid_request_error", "Body must be a JSON object"
        )
    user_id = getattr(user, "uid", None) or getattr(user, "user_id", "")
    return await audio_api.create_speech(body, user_id=user_id)


@router.post("/audio/transcriptions")
async def audio_transcriptions(
    user: UserData = Depends(auth),
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
    response_format: str = Form("json"),
) -> dict:
    """OpenAI-compatible transcriptions backed by Soniox."""
    user_id = getattr(user, "uid", None) or getattr(user, "user_id", "")
    content = await file.read()
    filename = file.filename or "audio.wav"
    return await audio_api.create_transcription(
        content,
        filename=filename,
        user_id=user_id,
        model=model,
        language=language,
        response_format=response_format,
    )


@router.post("/images/generations")
async def image_generations(
    request: Request,
    user: UserData = Depends(auth),
) -> object:
    """OpenAI-compatible image generation via OpenRouter."""
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise services.openai_error(
            400, "invalid_request_error", "Invalid JSON body"
        ) from exc
    if not isinstance(body, dict):
        raise services.openai_error(
            400, "invalid_request_error", "Body must be a JSON object"
        )
    user_id = getattr(user, "uid", None) or getattr(user, "user_id", "")
    return await images_api.create_generation(body, user_id=user_id)


@router.post("/images/edits")
async def image_edits(
    user: UserData = Depends(auth),
    file: UploadFile = File(...),
    prompt: str = Form(...),
    mask: UploadFile | None = File(None),
    model: str | None = Form(None),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("url"),
) -> object:
    """OpenAI-compatible image edits via OpenRouter."""
    user_id = getattr(user, "uid", None) or getattr(user, "user_id", "")
    image_bytes = await file.read()
    mask_bytes = await mask.read() if mask else None
    body = {
        "prompt": prompt,
        "model": model or "openai/dall-e-2",
        "n": n,
        "size": size,
        "response_format": response_format,
    }
    return await images_api.create_edit(
        body, user_id=user_id, image_bytes=image_bytes, mask_bytes=mask_bytes
    )
