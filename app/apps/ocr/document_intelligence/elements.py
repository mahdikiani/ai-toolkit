"""Element Processing — route each layout element to the right handler."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .layout import LayoutElement, LayoutType

logger = logging.getLogger(__name__)

TEXT_TYPES = {
    LayoutType.title, LayoutType.heading, LayoutType.paragraph,
    LayoutType.list, LayoutType.header, LayoutType.footer,
    LayoutType.reference, LayoutType.figure_caption,
}


@dataclass
class ProcessedElement:
    id: str
    page_id: str
    page_number: int
    type: LayoutType
    bbox: tuple[float, float, float, float]
    confidence: float

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
        vlm_model: str = "google/gemini-3.1-flash-lite",
        openrouter_client=None,
    ):
        self.vlm_model = vlm_model
        self.client = openrouter_client
        self.stats: list[dict] = []

    async def process(
        self, elem: LayoutElement, page_image: Image.Image
    ) -> ProcessedElement:
        crop_image = page_image.crop((
            int(elem.padded_bbox[0]), int(elem.padded_bbox[1]),
            int(elem.padded_bbox[2]), int(elem.padded_bbox[3]),
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
        result.id = elem.id
        result.page_id = elem.page_id
        result.page_number = elem.page_number
        result.type = elem.type
        result.bbox = elem.bbox
        result.confidence = elem.confidence

        self.stats.append({
            "id": elem.id,
            "type": elem.type.value,
            "duration": result.vlm_duration,
            "model": self.vlm_model,
        })
        return result

    async def _vlm_call(
        self, crop: Image.Image, system_prompt: str, user_prompt: str,
        response_format: dict | None = None, max_tokens: int = 1024,
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
            self.stats[-1]["tokens"] = data.get("usage", {}).get("total_tokens", 0)
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
        return content

    async def _process_text(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = "You are an OCR engine. Extract ALL text from this image exactly as written. Preserve original language. Use LaTeX for math. Return ONLY the extracted text."
        up = f"Extract the {elem.type.value} text from this crop."
        text = await self._vlm_call(crop, sp, up, max_tokens=2048)
        return ProcessedElement(text=text)

    async def _process_table(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = "You are a table extraction engine. Extract the table preserving ALL rows and columns exactly. Return HTML <table> format."
        up = "Extract this table. Return as HTML <table> with <tr> and <td> tags."
        html = await self._vlm_call(crop, sp, up, max_tokens=4096)
        return ProcessedElement(text=html, html=html)

    async def _process_formula(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = "You extract mathematical formulas. Return ONLY the LaTeX code. No explanations."
        up = "Extract the formula as LaTeX."
        latex = await self._vlm_call(crop, sp, up, max_tokens=1024)
        latex = latex.strip("`").strip().strip("$$").strip()
        return ProcessedElement(text=latex, latex=latex)

    async def _process_figure(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = "You describe images in documents. Provide: caption and description."
        up = "Describe this image. Format: caption: ...\\ndescription: ..."
        desc = await self._vlm_call(crop, sp, up, max_tokens=512)
        caption, description = desc, desc
        if "caption:" in desc.lower():
            parts = desc.split("description:", 1) if "description:" in desc.lower() else [desc]
            if len(parts) > 1:
                caption = parts[0].replace("caption:", "").strip()
                description = parts[1].strip()
        return ProcessedElement(caption=caption, description=description)

    async def _process_chart(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = """You extract chart information. Return JSON:
{"chart_type": "...", "title": "...", "description": "...", "data": [...]}"""
        up = "Extract this chart's information as JSON."
        import json as _json
        try:
            text = await self._vlm_call(
                crop, sp, up,
                response_format={"type": "json_object"},
                max_tokens=2048,
            )
            data = _json.loads(text)
        except Exception:
            text = await self._vlm_call(crop, sp, up, max_tokens=2048)
            data = {"chart_type": "unknown", "title": "", "description": text, "data": []}
        return ProcessedElement(
            text=data.get("description", ""),
            caption=data.get("title", ""),
            description=data.get("description", ""),
            chart_data=data,
        )

    async def _process_code(
        self, crop: Image.Image, elem: LayoutElement
    ) -> ProcessedElement:
        sp = "Extract code blocks exactly. Preserve indentation. Detect language."
        up = "Extract the code from this image."
        text = await self._vlm_call(crop, sp, up, max_tokens=2048)
        return ProcessedElement(text=text)

    def log_stats(self) -> None:
        if not self.stats:
            return
        by_type: dict[str, list[float]] = {}
        for s in self.stats:
            by_type.setdefault(s["type"], []).append(s["duration"])
        for t, ds in sorted(by_type.items()):
            logger.info(
                "  %s: %d elements, avg %.2fs", t, len(ds), sum(ds) / len(ds)
            )
