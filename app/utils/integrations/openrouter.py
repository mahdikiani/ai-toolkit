"""Shared OpenRouter HTTP client helpers (chat completions + SSE)."""

import asyncio
import json
import os
from asyncio import Queue, create_task
from collections.abc import AsyncIterator
from contextlib import suppress

import httpx

from server.config import Settings

OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "https://toolkit.uln.me")


class OpenRouterError(Exception):
    """Non-success HTTP status before a streaming body is returned."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Initialize the error with the provider status and detail."""
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class OpenRouterConfigurationError(ValueError):
    """Raised when an OpenRouter API key is not configured."""


class OpenRouterRequestError(RuntimeError):
    """Raised when an OpenRouter request fails."""


def chat_completions_url() -> str:
    """Full URL for POST /chat/completions."""
    return f"{Settings.openrouter_base_url}/chat/completions"


def resolve_api_key(explicit: str | None = None) -> str:
    """Return API key from argument or settings, or raise if missing."""
    key = explicit or Settings.openrouter_api_key
    if not key:
        error = OpenRouterConfigurationError("OpenRouter API key not configured")
        raise error
    return key


def build_headers(*, api_key: str | None = None) -> dict[str, str]:
    """Build authorization and JSON headers for OpenRouter."""
    return {
        "Authorization": f"Bearer {resolve_api_key(api_key)}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
    }


def extract_provider_meta(data: dict, *, provider: str) -> dict[str, object]:
    """Return normalized provider usage metadata from a completion response."""
    return {
        "provider": provider,
        "id": data.get("id"),
        "model": data.get("model"),
        "usage": data.get("usage"),
        "raw_cost": data.get("cost") or data.get("total_cost"),
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
        return chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
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
        error = OpenRouterRequestError(
            f"OpenRouter HTTP {e.response.status_code}: {e.response.text}"
        )
        raise error from e
    except httpx.RequestError as e:
        error = OpenRouterRequestError(f"OpenRouter request failed: {e}")
        raise error from e


async def _iter_chat_chunks(
    body: dict,
    *,
    api_key: str | None,
    http_timeout: float,
) -> AsyncIterator[dict]:
    """Yield parsed JSON chunks from a successful OpenRouter streaming request."""
    queue: Queue[dict | Exception | None] = Queue()

    async def produce() -> None:
        try:
            async with (
                httpx.AsyncClient() as client,
                client.stream(
                    "POST",
                    chat_completions_url(),
                    json={**body, "stream": True},
                    headers=build_headers(api_key=api_key),
                    timeout=http_timeout,
                ) as resp,
            ):
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        break
                    try:
                        await queue.put(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as error:
            await queue.put(error)
        finally:
            await queue.put(None)

    producer = create_task(produce())
    try:
        while (item := await queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer


async def stream_chat_deltas(
    body: dict,
    *,
    api_key: str | None = None,
    http_timeout: float = 120.0,
) -> AsyncIterator[str]:
    """Stream chat/completions and yield text content deltas."""
    async for delta, _usage in stream_chat_deltas_with_usage(
        body, api_key=api_key, http_timeout=http_timeout
    ):
        if delta:
            yield delta


async def stream_chat_deltas_with_usage(
    body: dict,
    *,
    api_key: str | None = None,
    http_timeout: float = 120.0,
) -> AsyncIterator[tuple[str, dict | None]]:
    """
    Stream chat/completions, yielding ``(text_delta, usage)`` per chunk.

    ``usage`` is ``None`` on every chunk except the terminal one, where
    OpenRouter includes a usage object by default (no ``stream_options``
    needed) -- callers that need to bill a streamed call for its real
    reported cost, not an estimate, should keep the last non-None usage
    seen across the iteration.
    """
    try:
        async for chunk_data in _iter_chat_chunks(
            body, api_key=api_key, http_timeout=http_timeout
        ):
            usage = chunk_data.get("usage")
            usage = usage if isinstance(usage, dict) else None
            choices = chunk_data.get("choices") or []
            content = ""
            if choices:
                content = (choices[0].get("delta") or {}).get("content") or ""
            if content or usage:
                yield content, usage
    except httpx.HTTPStatusError as e:
        error = OpenRouterRequestError(
            f"OpenRouter HTTP {e.response.status_code}: {e.response.text}"
        )
        raise error from e
    except httpx.RequestError as e:
        error = OpenRouterRequestError(f"OpenRouter request failed: {e}")
        raise error from e


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
    queue: Queue[bytes | Exception | None] = Queue()

    async def produce() -> None:
        try:
            async with (
                httpx.AsyncClient(timeout=http_timeout) as client,
                client.stream(
                    "POST",
                    chat_completions_url(),
                    json={**body, "stream": True},
                    headers=build_headers(api_key=api_key),
                ) as resp,
            ):
                if resp.status_code >= 400:
                    err = await resp.aread()
                    error = OpenRouterError(
                        resp.status_code, err.decode(errors="replace")
                    )
                    await queue.put(error)
                    return
                total_bytes = 0
                max_bytes = Settings.openai_stream_max_bytes
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        total_bytes += len(chunk)
                        if total_bytes > max_bytes:
                            error = OpenRouterError(
                                413,
                                f"Stream response exceeded {max_bytes} bytes",
                            )
                            await queue.put(error)
                            return
                        await queue.put(chunk)
        except Exception as error:
            await queue.put(error)
        finally:
            await queue.put(None)

    producer = create_task(produce())
    try:
        while (item := await queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        producer.cancel()
        with suppress(asyncio.CancelledError):
            await producer
