"""Services for promptic task management."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi_mongo_base.core.exceptions import BaseHTTPException
from fastapi_mongo_base.tasks import TaskStatusEnum

from server.config import Settings
from utils.billing import finance
from utils.integrations import openrouter as openrouter_client

from .engine import PromptEngine
from .engine.chunking import split_into_chunks

if TYPE_CHECKING:
    from .models import PrompticTask
    from .schemas import PrompticCreate

logger = logging.getLogger(__name__)

# Bound how many chunks of a long document are translated/processed at
# once -- fast enough for a whole book without hammering the provider
# with dozens of simultaneous requests.
_MAX_CONCURRENT_CHUNKS = 3

_GLOSSARY_SYSTEM_PROMPT = (
    "You extract a short glossary of recurring terms from a long document "
    "so that a later multi-part processing pass stays consistent across "
    "parts. List only terms that matter for consistency and recur "
    "multiple times: proper nouns, names, titles, technical terms. For "
    "each, give ONE fixed canonical rendering to use every time it "
    "appears. Output a compact list, one entry per line, formatted "
    "exactly as 'source -> canonical'. No headers, no commentary, no "
    "numbering."
)


def check_schemas(prompt_name: str, data: "PrompticCreate") -> None:
    """Validate that the prompt exists and input variables match schema."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{prompt_name}.yaml"

    if not prompt_path.exists():
        raise BaseHTTPException(
            status_code=404,
            error_code="prompt_not_found",
            detail=f"Prompt '{prompt_name}' not found",
            message={
                "en": f"Prompt '{prompt_name}' not found",
                "fa": f"پرامپت '{prompt_name}' یافت نشد",
            },
        )


async def call_openrouter(
    system: str,
    user: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    response_format: dict | None = None,
    return_meta: bool = False,
) -> str | tuple[str, dict[str, object]]:
    """Call OpenRouter API and optionally return provider metadata."""
    model = model or Settings.default_model
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    if response_format:
        body["response_format"] = response_format

    data = await openrouter_client.complete_chat_json(body, api_key=api_key)

    choices = data.get("choices")
    if not choices:
        raise RuntimeError(
            "No response from model; raw response: "
            + json.dumps(data, ensure_ascii=False)[:500]
        )
    content = choices[0].get("message", {}).get("content") or ""
    provider_meta = openrouter_client.extract_provider_meta(data, provider="openrouter")
    if not return_meta:
        return content.strip()
    return content.strip(), provider_meta


async def call_openrouter_stream(
    system: str,
    user: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
) -> AsyncIterator[str]:
    """Call OpenRouter API with streaming."""
    model = model or Settings.default_model
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens

    async for delta in openrouter_client.stream_chat_deltas(body, api_key=api_key):
        yield delta


def _cost_for(provider_meta: dict[str, object]) -> float:
    """Estimate the coin cost of one OpenRouter call from its provider_meta."""
    usage = provider_meta.get("usage")
    return finance.estimate_text_cost(
        model=str(provider_meta.get("model") or ""),
        usage=usage if isinstance(usage, dict) else None,
        raw_cost=_text_cost_value(provider_meta.get("raw_cost")),
    )


def _estimate_preflight_cost(model_cfg: dict, input_variables: dict) -> float:
    """
    Conservative pre-flight cost estimate, before any LLM call is made.

    Approximates tokens from character count (~4 chars/token) across all
    string input variables, assuming output length on the same order as
    input plus a margin for chunking overhead (glossary pass, per-chunk
    instructions). This overestimates compressive tasks (summarize,
    quiz) and is roughly right for content-preserving ones (translate,
    structure, notes) -- the right direction to err for a gate that
    runs before any money is spent.
    """
    chars = sum(len(v) for v in input_variables.values() if isinstance(v, str))
    approx_tokens = int(chars / 4) * 2  # input + output, roughly
    approx_tokens = int(approx_tokens * 1.2)  # chunking overhead margin
    return finance.estimate_text_cost(
        model=str(model_cfg.get("model") or ""),
        usage={"total_tokens": approx_tokens},
    )


async def _build_glossary(
    content: str, *, language: str | None, model: str | None
) -> tuple[str, dict[str, object]]:
    """
    Extract a short term glossary from the full document.

    Used as a pre-pass for chunked processing so the same name/term
    gets the same rendering in chunk 1 and chunk 30 of a long document
    -- unlike carrying context forward chunk-to-chunk, a shared
    glossary decided once up front doesn't let small drift compound
    over the length of a whole book.
    """
    target = f"Target rendering language: {language}.\n\n" if language else ""
    user = f"{target}---\n{content}\n---"
    result, provider_meta = await call_openrouter(
        _GLOSSARY_SYSTEM_PROMPT,
        user,
        model=model,
        max_tokens=2000,
        temperature=0.1,
        return_meta=True,
    )
    return result, provider_meta


def _chunk_instructions(index: int, total: int, glossary: str) -> str:
    """Build the addendum system instructions for one chunk of a long document."""
    parts = [
        (
            f"This document is being processed in {total} sequential parts "
            f"because it is too long for a single response. This is part "
            f"{index} of {total}."
        ),
        (
            "Output ONLY the corresponding content for this part -- no "
            "greeting, no introduction, no recap of earlier parts, no "
            "concluding remarks."
        ),
        "Do not repeat content already covered in earlier parts.",
    ]
    if glossary:
        parts.append(
            "Use these exact renderings for recurring terms, for consistency "
            f"across all parts of this document:\n{glossary}"
        )
    return "\n".join(parts)


async def _process_chunk(
    engine: PromptEngine,
    prompt_path: Path,
    task: "PrompticTask",
    model_cfg: dict,
    chunk: str,
    index: int,
    total: int,
    glossary: str,
) -> tuple[str, dict[str, object]]:
    """Run one chunk of a long document through the prompt template."""
    chunk_vars = {**task.input_variables, "content": chunk}
    system_prompt, user_prompt, response_format = engine.generate(
        prompt_path, chunk_vars
    )
    system_prompt = f"{system_prompt}\n\n{_chunk_instructions(index, total, glossary)}"
    return await call_openrouter(
        system_prompt,
        user_prompt,
        model=model_cfg.get("model"),
        max_tokens=model_cfg.get("max_tokens"),
        response_format=response_format,
        temperature=model_cfg.get("temperature", 0.2),
        return_meta=True,
    )


async def _process_chunked(
    engine: PromptEngine,
    prompt_path: Path,
    task: "PrompticTask",
    model_cfg: dict,
    content: str,
) -> tuple[str, dict[str, object]]:
    """
    Process a long document as ordered chunks, stitched back together.

    A single completion call has an output-token ceiling far below what
    a whole long document needs, no matter how high max_tokens is set --
    translating (or turning into study notes) an entire book requires
    many calls, one per chunk, run in original order.
    """
    chunk_max_chars = int(model_cfg["chunk_max_chars"])
    chunks = split_into_chunks(content, chunk_max_chars)
    total = len(chunks)

    glossary = ""
    chunk_metas: list[dict[str, object]] = []
    if model_cfg.get("use_glossary"):
        language = task.input_variables.get("language")
        glossary, glossary_meta = await _build_glossary(
            content,
            language=language if isinstance(language, str) else None,
            model=model_cfg.get("model"),
        )
        chunk_metas.append(glossary_meta)

    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_CHUNKS)

    async def _run(index: int, chunk: str) -> tuple[str, dict[str, object]]:
        async with semaphore:
            return await _process_chunk(
                engine, prompt_path, task, model_cfg, chunk, index + 1, total, glossary
            )

    chunk_results = await asyncio.gather(
        *(_run(i, chunk) for i, chunk in enumerate(chunks))
    )
    texts = [text for text, _meta in chunk_results]
    chunk_metas.extend(meta for _text, meta in chunk_results)

    provider_meta: dict[str, object] = {
        "chunked": True,
        "chunk_count": total,
        "chunks": chunk_metas,
    }
    return "\n\n".join(texts), provider_meta


async def invoke_stream(task: "PrompticTask") -> AsyncIterator[str]:
    """Execute prompt and stream the response."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{task.prompt_name}.yaml"

    if not prompt_path.exists():
        raise BaseHTTPException(
            status_code=404,
            error_code="prompt_not_found",
            detail=f"Prompt '{task.prompt_name}' not found",
            message={
                "en": f"Prompt '{task.prompt_name}' not found",
                "fa": f"پرامپت '{task.prompt_name}' یافت نشد",
            },
        )

    try:
        engine = PromptEngine(base_dir=prompts_dir)
        system_prompt, user_prompt, _ = engine.generate(
            prompt_path, task.input_variables
        )
        model_cfg = engine.load_model_config(prompt_path)

        full_response = ""
        async for chunk in call_openrouter_stream(
            system_prompt,
            user_prompt,
            model=model_cfg.get("model"),
            max_tokens=model_cfg.get("max_tokens"),
            temperature=model_cfg.get("temperature", 0.2),
        ):
            full_response += chunk
            yield chunk

        # Update task with final result
        task.result = full_response
        task.task_status = TaskStatusEnum.completed
        await task.save()

    except Exception as e:
        logger.exception("Error executing prompt")
        task.error = str(e)
        task.task_status = TaskStatusEnum.error
        await task.save()
        raise


async def process_promptic(
    task: "PrompticTask",
    *,
    force_restart: bool = False,
    sync: bool = False,
    **kwargs: object,
) -> "PrompticTask":
    """Process a promptic run by invoking the prompt template."""
    prompts_dir = Settings.prompts_dir
    prompt_path = prompts_dir / f"{task.prompt_name}.yaml"

    if not prompt_path.exists():
        await task.update_and_emit(
            error=f"Prompt '{task.prompt_name}' not found",
            task_status=TaskStatusEnum.error,
        )
        return task

    try:
        engine = PromptEngine(base_dir=prompts_dir)
        model_cfg = engine.load_model_config(prompt_path)

        # Gate on quota BEFORE spending anything -- a chunked job can be
        # dozens of calls (a whole book), so discovering insufficient
        # funds only after the fact would mean paying for work the user
        # can't afford and then failing to bill it.
        estimated = _estimate_preflight_cost(model_cfg, task.input_variables)
        quota = await finance.check_quota(
            task.user_id, estimated, raise_exception=False
        )
        if quota < estimated:
            await task.update_and_emit(
                error="insufficient_quota",
                task_status=TaskStatusEnum.error,
            )
            return task

        chunk_max_chars = model_cfg.get("chunk_max_chars")
        content = task.input_variables.get("content")

        if (
            chunk_max_chars
            and isinstance(content, str)
            and len(content) > int(chunk_max_chars)
        ):
            # Output must scale with input (translation, study notes, ...) --
            # a single completion call has an output ceiling far below what a
            # whole long document needs, so split and stitch chunks together.
            result, provider_meta = await _process_chunked(
                engine, prompt_path, task, model_cfg, content
            )
            amount = sum(_cost_for(m) for m in provider_meta.get("chunks", []))
        else:
            system_prompt, user_prompt, response_format = engine.generate(
                prompt_path, task.input_variables
            )
            openrouter_result = await call_openrouter(
                system_prompt,
                user_prompt,
                model=model_cfg.get("model"),
                max_tokens=model_cfg.get("max_tokens"),
                response_format=response_format,
                temperature=model_cfg.get("temperature", 0.2),
                return_meta=True,
            )
            if isinstance(openrouter_result, tuple):
                result, provider_meta = openrouter_result
            else:
                result = openrouter_result
                provider_meta = {}
            amount = _cost_for(provider_meta)

        # A billing-recording failure must never discard an already-computed
        # result -- the LLM cost was already spent, so the task still
        # completes for the user; the estimated amount is kept on the task
        # for later reconciliation even though it wasn't actually recorded.
        try:
            metered = await finance.meter_cost(
                task.user_id,
                amount,
                meta_data={
                    "service": "promptic",
                    "prompt": task.prompt_name,
                    "provider_meta": provider_meta,
                },
            )
        except Exception:
            logger.exception(
                "Failed to meter promptic usage for task %s", task.uid
            )
            metered = None

        await task.update_and_emit(
            result=result,
            provider_meta=provider_meta,
            usage_amount=float(metered.amount) if metered else amount,
            usage_id=metered.uid if metered else None,
            task_status=TaskStatusEnum.completed,
        )

    except Exception as e:
        logger.exception("Error executing prompt")
        await task.update_and_emit(
            error=str(e),
            task_status=TaskStatusEnum.error,
        )

    return task


# Backward-compatible alias used by older unit tests.
process_execution_task = process_promptic


def _text_cost_value(value: object) -> int | float | str | None:
    """Return a provider cost value accepted by the finance service."""
    if isinstance(value, (int, float, str)):
        return value
    return None
