"""Shared OpenRouter HTTP client helpers (chat completions + SSE)."""

import json
from collections.abc import AsyncIterator

import httpx

from server.config import Settings

OPENROUTER_HTTP_REFERER = "https://github.com/prompt-library"


class OpenRouterError(Exception):
    """Non-success HTTP status before a streaming body is returned."""

    def __init__(self, status_code: int, detail: str) -> None:  # noqa: D107
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def chat_completions_url() -> str:
    """Full URL for POST /chat/completions."""
    return f"{Settings.openrouter_base_url}/chat/completions"


def resolve_api_key(explicit: str | None = None) -> str:
    """Return API key from argument or settings, or raise if missing."""
    key = explicit or Settings.openrouter_api_key
    if not key:
        raise ValueError("OpenRouter API key not configured")
    return key


def build_headers(*, api_key: str | None = None) -> dict[str, str]:
    """Build authorization and JSON headers for OpenRouter."""
    return {
        "Authorization": f"Bearer {resolve_api_key(api_key)}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
    }


def parse_sse_delta_line(line: str) -> str | None:
    """Extract a text delta from one SSE line, or a stream sentinel."""
    if not line or line.startswith(":"):
        return None
    if line.startswith("data: "):
        line = line[6:]
    if line == "[DONE]":
        return "[DONE]"
    try:
        chunk_data = json.loads(line)
        return (
            chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
        )
    except json.JSONDecodeError:
        return None


async def complete_chat_json(
    body: dict,
    *,
    api_key: str | None = None,
    http_timeout: float = 120.0,
) -> dict:
    """POST chat/completions and return parsed JSON (raises on HTTP errors)."""
    url = chat_completions_url()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json=body,
                headers=build_headers(api_key=api_key),
                timeout=http_timeout,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"OpenRouter HTTP {e.response.status_code}: {e.response.text}"
        ) from e
    except httpx.RequestError as e:
        raise RuntimeError(f"OpenRouter request failed: {e}") from e


async def stream_chat_deltas(
    body: dict,
    *,
    api_key: str | None = None,
    http_timeout: float = 120.0,
) -> AsyncIterator[str]:
    """Stream chat/completions and yield text content deltas."""
    url = chat_completions_url()
    stream_body = {**body, "stream": True}
    try:
        async with httpx.AsyncClient() as client, client.stream(
            "POST",
            url,
            json=stream_body,
            headers=build_headers(api_key=api_key),
            timeout=http_timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                delta = parse_sse_delta_line(line)
                if delta == "[DONE]":
                    break
                if delta:
                    yield delta
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"OpenRouter HTTP {e.response.status_code}: {e.response.text}"
        ) from e
    except httpx.RequestError as e:
        raise RuntimeError(f"OpenRouter request failed: {e}") from e


async def post_chat_completion_unchecked(
    body: dict,
    *,
    api_key: str | None = None,
    http_timeout: float = 120.0,
) -> httpx.Response:
    """POST chat/completions without raising on non-2xx (for proxies)."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            chat_completions_url(),
            json=body,
            headers=build_headers(api_key=api_key),
            timeout=http_timeout,
        )


async def stream_chat_completion_bytes(
    body: dict,
    *,
    api_key: str | None = None,
    http_timeout: float = 120.0,
) -> AsyncIterator[bytes]:
    """Stream raw SSE bytes; raises OpenRouterError if the status is not OK."""
    stream_body = {**body, "stream": True}
    async with httpx.AsyncClient(timeout=http_timeout) as client, client.stream(
        "POST",
        chat_completions_url(),
        json=stream_body,
        headers=build_headers(api_key=api_key),
    ) as resp:
        if resp.status_code >= 400:
            err = await resp.aread()
            raise OpenRouterError(
                resp.status_code,
                err.decode(errors="replace"),
            )
        async for chunk in resp.aiter_bytes():
            if chunk:
                yield chunk
