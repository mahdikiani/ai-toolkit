"""Element Processing — route each layout element to the right handler."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from pathlib import Path

from PIL import Image

from .layout import LayoutElement, LayoutType

logger = logging.getLogger(__name__)

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

        if self.client:
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
            data = await self.client.complete_chat_json(body)
            content = data["choices"][0]["message"]["content"].strip()
            self._last_tokens = data.get("usage", {}).get("total_tokens", 0)
            return content

        from utils.integrations.openrouter import complete_chat_json

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
        data = await complete_chat_json(body)
        content = data["choices"][0]["message"]["content"].strip()
        self._last_tokens = data.get("usage", {}).get("total_tokens", 0)
        return content

    async def _process_text(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = _read_prompt("text")
        up = f"Extract the {elem.type.value} text from this crop."
        text = await self._vlm_call(crop, sp, up, max_tokens=2048)
        return ProcessedElement(text=text)

    async def _process_table(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = _read_prompt("table")
        up = "Extract this table. Return as HTML <table> with <tr> and <td> tags."
        html = await self._vlm_call(crop, sp, up, max_tokens=4096)
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
        caption, description = desc, desc
        if "caption:" in desc.lower():
            parts = (
                desc.split("description:", 1)
                if "description:" in desc.lower()
                else [desc]
            )
            if len(parts) > 1:
                caption = parts[0].replace("caption:", "").strip()
                description = parts[1].strip()
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
        return ProcessedElement(
            text=data.get("description", ""),
            caption=data.get("title", ""),
            description=data.get("description", ""),
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
