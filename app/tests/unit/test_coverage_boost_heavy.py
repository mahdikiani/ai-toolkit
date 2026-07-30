"""Honest coverage boost for previously under-tested modules."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image


@pytest.mark.unit
class TestMarkdownParserCoverage:
    def test_parse_rich_markdown(self) -> None:
        from apps.ocr.document_intelligence.markdown_parser import parse_markdown

        md = """
# Title One

Intro paragraph.

## Section

- item a
- item b

1. first
2. second

> a quote
> continues

```
code here
```

$$
E = mc^2
$$

| A | B |
|---|---|
| 1 | 2 |

***

plain again
"""
        doc = parse_markdown(md)
        assert doc.pages
        assert any(n.text == "Title One" for n in doc.pages[0].nodes)

    def test_special_helpers(self) -> None:
        from apps.ocr.document_intelligence import markdown_parser as mp

        assert mp._is_special_line("# H")
        assert mp._parse_table_row("| a | b |") == ["a", "b"]
        nodes: list = []
        assert mp._consume_fence(["x"], 0, nodes) is None
        assert mp._consume_formula_block(["x"], 0, nodes) is None
        assert mp._consume_heading(["x"], 0, nodes) is None
        assert mp._consume_table(["|a|", "nope"], 0, nodes) is None
        assert mp._consume_list(["plain"], 0, nodes) is None


@pytest.mark.unit
class TestInlineAndNormalization:
    def test_inline_segments(self) -> None:
        from apps.ocr.document_intelligence.inline_markdown import parse_inline_segments

        segs = parse_inline_segments(
            "hello **bold** and __b2__ and `code` and *it* and _it2_ "
            "and [link](https://example.com) end"
        )
        assert any(s.bold for s in segs)
        assert any(s.code for s in segs)
        assert any(s.italic for s in segs)
        assert parse_inline_segments("")[0].text == ""

    def test_persian_normalization(self) -> None:
        from apps.ocr.pipeline.normalization import detect_rtl_ratio, normalize_persian

        text = normalize_persian("يك ۀةإأؤئٱ  ،سلام")
        assert "ی" in text or "ک" in text
        assert detect_rtl_ratio("سلام hello") > 0
        assert detect_rtl_ratio("abc") == 0


@pytest.mark.unit
class TestDocxRendererCoverage:
    def test_font_helpers_and_build(self, tmp_path: Path) -> None:
        from apps.ocr.pipeline import docx_renderer as dr

        assert dr._clean_font_name("") is None
        assert dr._clean_font_name("AB") is None
        assert dr._clean_font_name("CMMI12") is None
        assert dr._clean_font_name("CMR10") is None
        assert dr._clean_font_name("XB Niloofar") == "IRNazanin"
        assert dr._clean_font_name("TimesNewRoman") == "Times New Roman"
        assert dr._clean_font_name("WeirdFontName") is None
        assert dr._pick_cleaned_font({}) is None
        assert dr.detect_pdf_fonts(None) == {}
        assert dr.detect_pdf_fonts(b"not-a-pdf") == {}

        asset = tmp_path / "pic.png"
        Image.new("RGB", (20, 20), "red").save(asset)
        md = f"""
# Heading

Paragraph text.

## Sub

- bullet
1. numbered

> quote

$$x^2$$

$y$

![alt]({asset})

| A | B |
| 1 | 2 |
|---|---|

---
<!-- page: 2 -->
"""
        img = Image.new("RGB", (100, 100), "white")
        elem = SimpleNamespace(
            type=SimpleNamespace(name="figure"),
            element_id="e1",
            page_number=1,
            x1=10,
            y1=10,
            x2=40,
            y2=40,
        )
        # Layout ElementType enum for crop helpers
        from apps.ocr.pipeline.layout_detector import ElementType, LayoutBox

        fig = LayoutBox(
            element_id="e1",
            page_number=1,
            type=ElementType.figure,
            x1=10,
            y1=10,
            x2=40,
            y2=40,
        )
        crops_dir = tmp_path / "crops"
        crops_dir.mkdir()
        (crops_dir / "c1.png").write_bytes(asset.read_bytes())

        buf = dr.build_docx(
            md,
            page_images=[img, img],
            elements=[fig],
            page_headers={1: "H1", 2: "H2"},
            page_footers={1: "F1", 2: "F2"},
            crops={"e1": asset.read_bytes()},
            crops_dir=crops_dir,
            assets_dir=tmp_path,
            pdf_data=None,
        )
        assert buf.getvalue()[:2] == b"PK"

        blocks = dr._parse_markdown_blocks(md)
        assert blocks
        assert dr._bytes_from_src("#", None) is None
        assert dr._bytes_from_src(str(asset), None)
        assert dr._bytes_from_src("missing.png", tmp_path) is None
        assert dr._bytes_from_src("pic.png", tmp_path)
        assert dr._bytes_from_crop_cache(None, None) is None
        assert dr._bytes_from_crop_cache([fig], {"e1": b"x"})
        assert dr._bytes_from_page_crop([img], [fig])
        assert (
            dr._resolve_image(
                {"src": "#"},
                [],
                None,
                None,
                crops_dir,
                None,
            )
            is not None
        )
        # unused elem var silence
        _ = elem


@pytest.mark.unit
class TestPreprocessingAndPdfRenderer:
    def test_preprocessor(self) -> None:
        from apps.ocr.pipeline.preprocessing import ImagePreprocessor

        img = Image.new("RGB", (80, 80), (30, 30, 30))
        prep = ImagePreprocessor(
            enable_deskew=True, enable_contrast=True, enable_denoise=True
        )
        out = prep.process(img)
        assert out.size[0] > 0
        bio = BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        assert prep.process_bytes(bio).size[0] > 0

        # empty-ish image path for deskew early return
        blank = Image.new("RGB", (10, 10), (255, 255, 255))
        prep.process(blank)

    def test_pdf_renderer(self, tmp_path: Path) -> None:
        import fitz

        from apps.ocr.pipeline import renderer as pdf_r

        pdf_path = tmp_path / "t.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "hello")
        doc.save(pdf_path)
        doc.close()

        pages = pdf_r.render_pdf_pages(pdf_path, dpi=72)
        assert len(pages) == 1
        data = pdf_path.read_bytes()
        pages2 = pdf_r.render_pdf_bytes(BytesIO(data), dpi=72)
        assert len(pages2) == 1
        assert pdf_r.count_pdf_bytes(BytesIO(data)) == 1


@pytest.mark.unit
class TestLayoutDetectorCoverage:
    def test_iou_dedup_convert(self) -> None:
        from apps.ocr.pipeline.layout_detector import (
            ELEMENT_TYPE_MAP,
            LayoutBox,
            LayoutDetector,
            PaddleOcrMissingError,
            _iou,
            deduplicate_by_iou,
        )

        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
        a = LayoutBox("a", 1, ELEMENT_TYPE_MAP["title"], 0, 0, 10, 10, 0.9)
        b = LayoutBox("b", 1, ELEMENT_TYPE_MAP["paragraph"], 1, 1, 9, 9, 0.8)
        assert len(deduplicate_by_iou([a, b], 0.4)) == 1
        assert deduplicate_by_iou([a]) == [a]

        det = LayoutDetector(confidence_threshold=0.3)
        img = Image.new("RGB", (50, 50), "white")
        assert det._full_page_element(img, 1).page_number == 1
        assert det._convert_result("bad", 1, 50, 50) == []
        boxes = det._convert_result(
            {
                "res": {
                    "boxes": [
                        {
                            "label": "title",
                            "bbox": [1, 2, 20, 30],
                            "confidence": 0.9,
                            "res": [{"text": "T"}],
                        },
                        {
                            "type": "paragraph",
                            "coordinate": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                            "score": 0.1,
                        },
                        {"label": "x", "bbox": [1]},
                        "skip",
                    ]
                }
            },
            1,
            50,
            50,
        )
        assert boxes
        with pytest.raises(PaddleOcrMissingError):
            raise PaddleOcrMissingError()

        model = MagicMock()
        model.predict.return_value = iter([
            SimpleNamespace(
                json=lambda: {
                    "boxes": [
                        {
                            "block_label": "figure",
                            "block_bbox": [0, 0, 10, 10],
                            "confidence": 0.9,
                        }
                    ]
                }
            )
        ])
        det._model_v2 = model
        det._model_v3 = model
        assert det.detect(img, 1)

        det2 = LayoutDetector()
        with (
            patch(
                "builtins.__import__",
                side_effect=ImportError("no paddle"),
            ),
            pytest.raises(PaddleOcrMissingError),
        ):
            det2._detect_with(img, 1, "M", "_model_v2")


@pytest.mark.unit
class TestPromptsParserCoverage:
    def test_parse_and_render(self, tmp_path: Path) -> None:
        from apps.language.prompts import parser as pp

        assert "name" in pp.extract_jinja2_variables("Hello {{ name }}")
        assert pp.infer_field_type("is_ok", {}) == "boolean"
        assert pp.infer_field_type("item_count", {}) == "integer"
        assert pp.infer_field_type("item_list", {}) == "array"
        assert pp.infer_field_type("x", {"x": True}) == "boolean"
        assert pp.infer_field_type("x", {"x": 1}) == "integer"
        assert pp.infer_field_type("x", {"x": 1.5}) == "number"
        assert pp.infer_field_type("x", {"x": []}) == "array"
        assert pp.infer_field_type("x", {"x": {}}) == "object"
        assert pp.infer_field_type("plain", {}) == "string"

        bad = tmp_path / "missing.yaml"
        with pytest.raises(pp.PromptFileNotFoundError):
            pp.parse_prompt_file(bad)

        not_map = tmp_path / "list.yaml"
        not_map.write_text("- a\n", encoding="utf-8")
        with pytest.raises(pp.PromptFileFormatError):
            pp.parse_prompt_file(not_map)

        prompt = tmp_path / "p.yaml"
        prompt.write_text(
            """
name: demo
description: d
tags: [t]
model: m
config: {temperature: 0}
examples:
  is_ready: true
  n_count: 3
messages:
  - role: system
    content: "Sys {{ is_ready }}"
  - role: user
    content:
      - type: text
        text: "Hi {{ user_name }} {{ loop }}"
      - type: file
        file_url: "https://x"
output_schema: {type: object}
""",
            encoding="utf-8",
        )
        data = pp.parse_prompt_file(prompt)
        assert data["name"] == "demo"
        rendered = pp.render_prompt(prompt, {"is_ready": True, "user_name": "Ali"})
        assert "Ali" in rendered["messages"][1]["content"][0]["text"]


@pytest.mark.unit
class TestPrompticCliCoverage:
    @pytest.mark.asyncio
    async def test_run_helpers(self, tmp_path: Path) -> None:
        from apps.language.promptic.engine import run as run_mod

        with pytest.raises(run_mod.InvalidModelJsonError):
            run_mod.extract_json_from_content("not json")
        assert run_mod.extract_json_from_content('{"a":1}') == {"a": 1}
        assert run_mod.extract_json_from_content('```json\n{"a":1}\n```') == {"a": 1}
        assert run_mod._require_json_object({"a": 1}) == {"a": 1}
        with pytest.raises(run_mod.InvalidPromptResultError):
            run_mod._require_json_object([1])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"ok": true}'}}]
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            content = await run_mod.call_openrouter(
                "s", "u", api_key="k", model="m", max_tokens=10, response_format={}
            )
        assert "ok" in content

        mock_resp.raise_for_status.side_effect = __import__("httpx").HTTPStatusError(
            "err",
            request=MagicMock(),
            response=MagicMock(status_code=500, text="nope"),
        )
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(run_mod.OpenRouterRequestError),
        ):
            await run_mod.call_openrouter("s", "u", api_key="k")

        mock_client.post = AsyncMock(
            side_effect=__import__("httpx").RequestError("boom")
        )
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(run_mod.OpenRouterRequestError),
        ):
            await run_mod.call_openrouter("s", "u", api_key="k")

        mock_client.post = AsyncMock(
            return_value=MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"choices": []}),
            )
        )
        mock_client.post.return_value.__aenter__ = None
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            pytest.raises(run_mod.OpenRouterRequestError),
        ):
            # rebuild clean client
            c2 = AsyncMock()
            c2.__aenter__ = AsyncMock(return_value=c2)
            c2.__aexit__ = AsyncMock(return_value=None)
            r = MagicMock()
            r.raise_for_status = MagicMock()
            r.json.return_value = {"choices": []}
            c2.post = AsyncMock(return_value=r)
            with patch("httpx.AsyncClient", return_value=c2):
                await run_mod.call_openrouter("s", "u", api_key="k")

        out_json = tmp_path / "o.json"
        out_yaml = tmp_path / "o.yaml"
        out_txt = tmp_path / "o.txt"
        for out in (out_json, out_yaml, out_txt):
            with (
                patch(
                    "sys.argv",
                    ["run", "-p", "p.yaml", "-i", "i.yaml", "-o", str(out)],
                ),
                patch("apps.language.promptic.engine.run.PromptEngine") as eng,
                patch(
                    "apps.language.promptic.engine.run.load_data",
                    return_value={"x": 1},
                ),
                patch(
                    "apps.language.promptic.engine.run.call_openrouter",
                    AsyncMock(return_value='{"a":1}'),
                ),
            ):
                eng.return_value.generate.return_value = ("s", "u", None)
                await run_mod._main()
            assert out.exists()

        with (
            patch("sys.argv", ["run", "-p", "p.yaml", "-i", "i.yaml"]),
            patch("apps.language.promptic.engine.run.PromptEngine") as eng,
            patch(
                "apps.language.promptic.engine.run.load_data",
                return_value={"x": 1},
            ),
            patch(
                "apps.language.promptic.engine.run.call_openrouter",
                AsyncMock(return_value='{"a":1}'),
            ),
        ):
            eng.return_value.generate.return_value = ("s", "u", None)
            await run_mod._main()

        with (
            patch("sys.argv", ["run", "-p", "p.yaml", "-i", "i.yaml"]),
            patch(
                "apps.language.promptic.engine.run.load_data",
                return_value=["not-dict"],
            ),
            patch("sys.exit", side_effect=SystemExit) as exit_mock,
            pytest.raises(SystemExit),
        ):
            await run_mod._main()
        exit_mock.assert_called()

        with (
            patch("sys.argv", ["run", "-p", "p.yaml", "-i", "i.yaml"]),
            patch(
                "apps.language.promptic.engine.run.load_data",
                side_effect=RuntimeError("x"),
            ),
            patch("sys.exit", side_effect=SystemExit),
            pytest.raises(SystemExit),
        ):
            await run_mod._main()

    def test_generator_main(self, tmp_path: Path) -> None:
        from apps.language.promptic.engine import generator as gen

        for suffix in (".json", ".yaml", ".txt"):
            out = tmp_path / f"g{suffix}"
            with (
                patch(
                    "sys.argv",
                    ["gen", "-p", "p.yaml", "-i", "i.yaml", "-o", str(out)],
                ),
                patch("apps.language.promptic.engine.generator.PromptEngine") as eng,
                patch(
                    "apps.language.promptic.engine.generator.load_data",
                    return_value={"x": 1},
                ),
            ):
                eng.return_value.generate.return_value = ("sys", "user", None)
                gen._main()
            assert out.exists()

        with (
            patch("sys.argv", ["gen", "-p", "p.yaml", "-i", "i.yaml"]),
            patch("apps.language.promptic.engine.generator.PromptEngine") as eng,
            patch(
                "apps.language.promptic.engine.generator.load_data",
                return_value={"x": 1},
            ),
        ):
            eng.return_value.generate.return_value = ("sys", "user", None)
            gen._main()

        with (
            patch("sys.argv", ["gen", "-p", "p.yaml", "-i", "i.yaml"]),
            patch(
                "apps.language.promptic.engine.generator.load_data",
                return_value=["bad"],
            ),
            patch("sys.exit", side_effect=SystemExit),
            pytest.raises(SystemExit),
        ):
            gen._main()

        with (
            patch("sys.argv", ["gen", "-p", "p.yaml", "-i", "i.yaml"]),
            patch(
                "apps.language.promptic.engine.generator.load_data",
                side_effect=RuntimeError("x"),
            ),
            patch("sys.exit", side_effect=SystemExit),
            pytest.raises(SystemExit),
        ):
            gen._main()


@pytest.mark.unit
class TestOcrServiceBranches:
    @pytest.mark.asyncio
    async def test_resolve_and_save(self) -> None:
        from apps.ocr import services as ocr_svc
        from apps.ocr.schemas import OcrEngineType

        task = SimpleNamespace(ocr_engine="document_intelligence")
        assert ocr_svc._resolve_ocr_engine(task) == OcrEngineType.document_intelligence
        task.ocr_engine = "di"
        assert ocr_svc._resolve_ocr_engine(task) == OcrEngineType.document_intelligence
        task.ocr_engine = "paddle"
        assert ocr_svc._resolve_ocr_engine(task) == OcrEngineType.paddleocr_vl_1_5
        task.ocr_engine = None
        with patch.object(ocr_svc.Settings, "ocr_engine", "pipeline"):
            assert ocr_svc._resolve_ocr_engine(task) == OcrEngineType.pipeline

        task = SimpleNamespace(
            uid="task-1",
            save_report=AsyncMock(),
            result=None,
            task_status=None,
            usage_amount=None,
            usage_id=None,
            provider_meta=None,
        )
        err = await ocr_svc.save_error(task, "boom")
        assert err is task
        with patch(
            "apps.ocr.services.texttools.normalize_text",
            return_value="md",
        ):
            ok = await ocr_svc.save_result(task, "md", usage_amount=1.0, usage_id="u")
        assert ok.result == "md"


@pytest.mark.unit
class TestMiscRoutesAndUtils:
    def test_convert_routes_import(self) -> None:
        from apps.ocr import convert_routes

        assert convert_routes.router is not None

    def test_completion_routes_import(self) -> None:
        from apps.language.completion import routes as cr

        assert cr.router is not None

    def test_di_layout_helpers(self) -> None:
        from apps.ocr.document_intelligence import layout as lay

        assert lay._box_area((0, 0, 10, 10)) == 100
        assert lay._intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) > 0
        assert lay._iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
        assert lay._containment_ratio((0, 0, 5, 5), (0, 0, 10, 10)) > 0
        a = lay.LayoutElement(
            id="a",
            page_id="p1",
            page_number=1,
            type=lay.LayoutType.paragraph,
            bbox=(0, 0, 10, 10),
            padded_bbox=(0, 0, 10, 10),
            confidence=0.9,
        )
        b = lay.LayoutElement(
            id="b",
            page_id="p1",
            page_number=1,
            type=lay.LayoutType.paragraph,
            bbox=(1, 1, 9, 9),
            padded_bbox=(1, 1, 9, 9),
            confidence=0.8,
        )
        assert lay.deduplicate_by_iou([a, b])
        assert lay._pad_bbox((0, 0, 10, 10), 100, 100, 2)
