"""Element Processing — route each layout element to the right handler."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from pathlib import Path

from PIL import Image

from ..pipeline.normalization import normalize_persian, normalize_persian_digits
from .layout import LayoutElement, LayoutType

logger = logging.getLogger(__name__)

# A document can be hundreds of pages / thousands of VLM calls -- over
# that many calls, a transient failure (rate limit, network blip,
# provider hiccup) is expected to happen at least once. Without retry,
# any single element's failure kills the entire multi-hour job with
# nothing salvaged, which is why retry lives at the call level rather
# than only at the page/document level.
_MAX_VLM_RETRIES = 3
_VLM_RETRY_BACKOFF_SECONDS = 2.0

TEXT_TYPES = {
    LayoutType.title,
    LayoutType.heading,
    LayoutType.paragraph,
    LayoutType.list,
    LayoutType.header,
    LayoutType.footer,
    LayoutType.reference,
    LayoutType.figure_caption,
}

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@cache
def _read_prompt(name: str) -> str:
    """Read a VLM system prompt from prompts/<name>.prompt."""
    return (_PROMPTS_DIR / f"{name}.prompt").read_text().strip()


# A real caption ("Figure 1", a short Persian label) is a short label,
# not a sentence -- the VLM is asked for "a short one-line caption" but
# doesn't always comply, sometimes labeling a full descriptive sentence
# as the caption. Past this length it's treated as unreliable and folded
# into the description instead, rather than trusted to render visibly.
_MAX_CAPTION_CHARS = 80


def _split_caption_description(text: str) -> tuple[str, str]:
    r"""
    Split a VLM "caption: ...\ndescription: ..." response into its parts.

    Falls back to an *empty* caption with the whole response as the
    description when the VLM doesn't follow the requested format, or
    when it does but "caption" came back too long to plausibly be one
    (see _MAX_CAPTION_CHARS) -- caption is rendered as real visible text
    in the output document (unlike description, which becomes image alt
    text/accessibility metadata), so a caption that's really a
    description in disguise must never leak into it. Matched
    case-insensitively (a previous version matched "description:"
    case-insensitively but then sliced the original-cased text with it,
    so an actual "Description:" response silently failed to split and
    hit this same leak).
    """
    lower = text.lower()
    caption_idx = lower.find("caption:")
    description_idx = lower.find("description:")
    if 0 <= caption_idx < description_idx:
        caption = text[caption_idx + len("caption:") : description_idx].strip()
        description = text[description_idx + len("description:") :].strip()
        if len(caption) <= _MAX_CAPTION_CHARS:
            return caption, description
        return "", f"{caption} {description}".strip()
    return "", text.strip()


@dataclass
class ProcessedElement:
    """
    Result of processing one layout element through a VLM handler.

    All fields default so handlers can build a ProcessedElement with only
    the content fields set; process() fills id/page_id/.../confidence in
    from the source LayoutElement right after the handler returns.
    """

    id: str = ""
    page_id: str = ""
    page_number: int = 0
    type: LayoutType = LayoutType.unknown
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confidence: float = 0.0

    text: str = ""
    html: str = ""  # for tables: HTML representation
    latex: str = ""  # for formulas
    caption: str = ""  # for figures/charts
    description: str = ""  # for figures/charts
    chart_data: dict | None = None  # for charts
    asset_path: str = ""  # for figures/charts

    vlm_model: str = ""
    vlm_duration: float = 0.0
    vlm_tokens: int = 0


class ElementProcessor:
    """Process each layout element using the appropriate VLM strategy."""

    def __init__(
        self,
        vlm_model: str | None = None,
        openrouter_client: object | None = None,
        max_concurrent: int = 5,
    ) -> None:
        """Configure the VLM model and optional injected client for tests."""
        if vlm_model is None:
            from server.config import Settings

            vlm_model = Settings.ocr_vlm_model
        self.vlm_model = vlm_model
        self.client = openrouter_client
        self.max_concurrent = max_concurrent
        self.stats: list[dict] = []
        self._last_tokens = 0

    async def process(
        self, elem: LayoutElement, page_image: Image.Image
    ) -> ProcessedElement:
        """Route *elem* to the matching VLM handler and record stats."""
        crop_image = page_image.crop((
            int(elem.padded_bbox[0]),
            int(elem.padded_bbox[1]),
            int(elem.padded_bbox[2]),
            int(elem.padded_bbox[3]),
        ))
        t0 = time.time()

        if elem.type in TEXT_TYPES:
            result = await self._process_text(crop_image, elem)
        elif elem.type == LayoutType.table:
            result = await self._process_table(crop_image, elem)
        elif elem.type == LayoutType.formula:
            result = await self._process_formula(crop_image, elem)
        elif elem.type == LayoutType.figure:
            result = await self._process_figure(crop_image, elem)
        elif elem.type == LayoutType.chart:
            result = await self._process_chart(crop_image, elem)
        elif elem.type == LayoutType.code:
            result = await self._process_code(crop_image, elem)
        else:
            result = await self._process_text(crop_image, elem)

        result.vlm_duration = time.time() - t0
        result.vlm_model = self.vlm_model
        result.vlm_tokens = self._last_tokens
        result.id = elem.id
        result.page_id = elem.page_id
        result.page_number = elem.page_number
        result.type = elem.type
        result.bbox = elem.bbox
        result.confidence = elem.confidence

        self.stats.append({
            "id": elem.id,
            "type": elem.type.value,
            "confidence": elem.confidence,
            "duration": result.vlm_duration,
            "tokens": result.vlm_tokens,
            "model": self.vlm_model,
        })
        return result

    @staticmethod
    async def _call_with_retry(
        call_fn: Callable[[dict], Awaitable[dict]], body: dict
    ) -> dict:
        """Call a complete_chat_json-shaped function with retry + backoff."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_VLM_RETRIES):
            try:
                return await call_fn(body)
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_VLM_RETRIES - 1:
                    delay = _VLM_RETRY_BACKOFF_SECONDS * (2**attempt)
                    logger.warning(
                        "VLM call failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        _MAX_VLM_RETRIES,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
        assert last_exc is not None  # ruff: ignore[assert] -- loop always sets it before exit
        raise last_exc

    async def _vlm_call(
        self,
        crop: Image.Image,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        buf = BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        import base64

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        body = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if response_format:
            body["response_format"] = response_format

        if self.client:
            call_fn = self.client.complete_chat_json
        else:
            from utils.integrations.openrouter import complete_chat_json as call_fn

        data = await self._call_with_retry(call_fn, body)
        content = data["choices"][0]["message"]["content"].strip()
        self._last_tokens = data.get("usage", {}).get("total_tokens", 0)
        return content

    async def _process_text(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = _read_prompt("text")
        up = f"Extract the {elem.type.value} text from this crop."
        text = await self._vlm_call(crop, sp, up, max_tokens=2048)
        return ProcessedElement(text=normalize_persian(text))

    async def _process_table(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = _read_prompt("table")
        up = "Extract this table. Return as HTML <table> with <tr> and <td> tags."
        html = await self._vlm_call(crop, sp, up, max_tokens=4096)
        # Digits only here, not the full normalize_persian() -- its
        # spacing/punctuation passes aren't safe to run over raw HTML tag
        # syntax.
        html = normalize_persian_digits(html)
        return ProcessedElement(text=html, html=html)

    async def _process_formula(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = _read_prompt("formula")
        up = "Extract the formula as LaTeX."
        latex = await self._vlm_call(crop, sp, up, max_tokens=1024)
        latex = latex.strip("`").strip().strip("$").strip()
        return ProcessedElement(text=latex, latex=latex)

    async def _process_figure(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = _read_prompt("figure")
        up = "Describe this image. Format: caption: ...\\ndescription: ..."
        desc = await self._vlm_call(crop, sp, up, max_tokens=512)
        caption, description = _split_caption_description(normalize_persian(desc))
        return ProcessedElement(caption=caption, description=description)

    async def _process_chart(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = _read_prompt("chart")
        up = "Extract this chart's information as JSON."
        import json as _json

        try:
            text = await self._vlm_call(
                crop,
                sp,
                up,
                response_format={"type": "json_object"},
                max_tokens=2048,
            )
            data = _json.loads(text)
        except Exception:
            text = await self._vlm_call(crop, sp, up, max_tokens=2048)
            data = {
                "chart_type": "unknown",
                "title": "",
                "description": text,
                "data": [],
            }
        # Only title/description are shown as text -- chart_data's own
        # numeric "data" array is left untouched so chart values stay
        # exact.
        title = normalize_persian(data.get("title", ""))
        description = normalize_persian(data.get("description", ""))
        return ProcessedElement(
            text=description,
            caption=title,
            description=description,
            chart_data=data,
        )

    async def _process_code(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = _read_prompt("code")
        up = "Extract the code from this image."
        text = await self._vlm_call(crop, sp, up, max_tokens=2048)
        return ProcessedElement(text=text)

    def log_stats(self) -> None:
        """Log per-type average processing duration."""
        if not self.stats:
            return
        by_type: dict[str, list[float]] = {}
        for s in self.stats:
            by_type.setdefault(s["type"], []).append(s["duration"])
        for t, ds in sorted(by_type.items()):
            logger.info("  %s: %d elements, avg %.2fs", t, len(ds), sum(ds) / len(ds))
