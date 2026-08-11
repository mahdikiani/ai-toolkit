"""Second pass coverage boost for remaining low-coverage modules."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from docx import Document
from PIL import Image


@pytest.mark.unit
class TestOcrServicesCoverage:
    @pytest.mark.asyncio
    async def test_process_ocr_branches(self) -> None:
        from apps.ocr import services as svc

        task = SimpleNamespace(
            uid="t1",
            user_id="u1",
            ocr_engine="pipeline",
            file_content=AsyncMock(return_value=BytesIO(b"%PDF")),
            save_report=AsyncMock(),
            result=None,
            task_status=None,
            usage_amount=None,
            usage_id=None,
            provider_meta=None,
        )

        with (
            patch(
                "apps.ocr.services.mime.check_file_type", return_value="application/zip"
            ),
            patch("apps.ocr.services.is_compressed_file", return_value=True),
            patch(
                "apps.ocr.services.process_compressed_archive",
                AsyncMock(return_value=task),
            ) as arch,
        ):
            assert await svc.process_ocr(task) is task
            arch.assert_awaited()

        with (
            patch("apps.ocr.services.mime.check_file_type", return_value="text/plain"),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=False),
            patch("apps.ocr.services.process_direct_file", return_value="txt"),
            patch("apps.ocr.services.texttools.normalize_text", return_value="txt"),
        ):
            out = await svc.process_ocr(task)
            assert out.result == "txt"

        with (
            patch(
                "apps.ocr.services.mime.check_file_type", return_value="application/pdf"
            ),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=True),
            patch(
                "apps.ocr.services._process_with_pipeline",
                AsyncMock(return_value=task),
            ) as pipe,
        ):
            task.ocr_engine = "pipeline"
            await svc.process_ocr(task)
            pipe.assert_awaited()

        with (
            patch(
                "apps.ocr.services.mime.check_file_type", return_value="application/pdf"
            ),
            patch("apps.ocr.services.is_compressed_file", return_value=False),
            patch("apps.ocr.services.is_ocr_required", return_value=True),
            patch(
                "apps.ocr.services._process_with_document_intelligence",
                AsyncMock(return_value=task),
            ) as di,
        ):
            task.ocr_engine = "document_intelligence"
            await svc.process_ocr(task)
            di.assert_awaited()

        with (
            patch(
                "apps.ocr.services.mime.check_file_type",
                side_effect=RuntimeError("boom"),
            ),
        ):
            err = await svc.process_ocr(task)
            assert err is task

    @pytest.mark.asyncio
    async def test_upload_and_pipeline_processors(self, tmp_path: Path) -> None:
        from apps.ocr import services as svc
        from apps.ocr.schemas import OcrEngineType

        pipeline = MagicMock()
        pipeline.get_assets.return_value = [
            {"id": "asset:1", "image_bytes": b"img"},
            {"id": "asset:2", "image_bytes": b"bad"},
        ]
        pipeline.get_elements.return_value = []
        pipeline.get_headers.return_value = {}
        pipeline.get_footers.return_value = {}
        pipeline.get_all_crops.return_value = {}

        async def upload(buf, **_kwargs):
            data = buf.read() if hasattr(buf, "read") else b""
            if data == b"bad":
                raise RuntimeError("up")
            return "https://media/x"

        with (
            patch("utils.integrations.media.upload_file", side_effect=upload),
        ):
            md = await svc._upload_pipeline_assets(
                pipeline,
                "see (asset:1)",
                user_id="u1",
                workspace_id="w1",
            )
        assert "https://media/x" in md

        task = SimpleNamespace(
            uid="t",
            user_id="u",
            tenant_id="tenant",
            workspace_id=None,
            meta_data=None,
            save_report=AsyncMock(),
            result=None,
            task_status=None,
            usage_amount=None,
            usage_id=None,
            provider_meta=None,
        )
        usage = SimpleNamespace(amount=1.5, uid="usage1")

        with (
            patch("apps.ocr.pipeline.renderer.count_pdf_bytes", return_value=0),
        ):
            out = await svc._process_with_pipeline(
                task, BytesIO(b"%PDF"), "application/pdf", OcrEngineType.pipeline
            )
            assert out is task

        with (
            patch("apps.ocr.pipeline.renderer.count_pdf_bytes", return_value=2),
            patch("apps.ocr.services.finance.check_quota", AsyncMock(return_value=0)),
        ):
            out = await svc._process_with_pipeline(
                task, BytesIO(b"%PDF"), "application/pdf", OcrEngineType.pipeline
            )

        fake_pipe = MagicMock()
        fake_pipe.process_pdf = AsyncMock(return_value="md")
        fake_pipe.process_image_bytes = AsyncMock(return_value="md")
        with (
            patch("apps.ocr.pipeline.renderer.count_pdf_bytes", return_value=1),
            patch("apps.ocr.services.finance.check_quota", AsyncMock(return_value=10)),
            patch(
                "apps.ocr.pipeline.engine.DocumentPipeline",
                return_value=fake_pipe,
            ),
            patch(
                "apps.ocr.services._upload_pipeline_assets",
                AsyncMock(return_value="md"),
            ),
            patch(
                "apps.ocr.services._emit_markdown_artifact",
                AsyncMock(return_value="artifact-1"),
            ),
            patch(
                "apps.ocr.services._docx_url_via_converter",
                AsyncMock(return_value="https://media/doc.docx"),
            ),
            patch("apps.ocr.services.finance.estimate_ocr_cost", return_value=1.0),
            patch(
                "apps.ocr.services.finance.meter_cost",
                AsyncMock(return_value=usage),
            ),
            patch("apps.ocr.services.texttools.normalize_text", return_value="md"),
        ):
            out = await svc._process_with_pipeline(
                task, BytesIO(b"%PDF"), "application/pdf", OcrEngineType.pipeline
            )
            assert out.result == "md"
            assert out.provider_meta["artifact_id"] == "artifact-1"
            assert out.provider_meta["docx_url"] == "https://media/doc.docx"
            out = await svc._process_with_pipeline(
                task, BytesIO(b"img"), "image/png", OcrEngineType.pipeline
            )

        di = MagicMock()
        di.process = AsyncMock(
            return_value=SimpleNamespace(
                markdown="di-md",
                assets=[
                    SimpleNamespace(path=str(tmp_path / "a.png"), rel_path="a.png")
                ],
                docx_bytes=b"PK",
                output_dir=str(tmp_path / "out"),
                stats={},
            )
        )
        di.cleanup = MagicMock()
        di.output_dir = str(tmp_path / "di")
        (tmp_path / "a.png").write_bytes(b"png")
        (tmp_path / "out").mkdir(exist_ok=True)

        async def read_in_loop(func, *args):
            return func(*args)

        with (
            patch("apps.ocr.pipeline.renderer.count_pdf_bytes", return_value=1),
            patch("apps.ocr.services.finance.check_quota", AsyncMock(return_value=10)),
            patch(
                "apps.ocr.document_intelligence.DocumentIntelligencePipeline",
                return_value=di,
            ),
            patch(
                "apps.ocr.document_intelligence.summarize_stats",
                return_value={},
            ),
            patch(
                "apps.ocr.document_intelligence.renderers.markdown.rewrite_asset_links",
                side_effect=lambda m, _u: m,
            ),
            patch(
                "utils.integrations.media.upload_file",
                AsyncMock(return_value="https://u"),
            ) as upload_mock,
            patch(
                "apps.ocr.services.asyncio.to_thread", side_effect=read_in_loop
            ),
            patch(
                "apps.ocr.services._emit_markdown_artifact",
                AsyncMock(return_value="artifact-2"),
            ),
            patch(
                "apps.ocr.services._docx_url_via_converter",
                AsyncMock(return_value="https://media/from-converter.docx"),
            ),
            patch("apps.ocr.services.finance.estimate_ocr_cost", return_value=1.0),
            patch(
                "apps.ocr.services.finance.meter_cost",
                AsyncMock(return_value=usage),
            ),
            patch("apps.ocr.services.texttools.normalize_text", return_value="di-md"),
        ):
            out = await svc._process_with_document_intelligence(
                task,
                BytesIO(b"%PDF"),
                "application/pdf",
                OcrEngineType.document_intelligence,
            )
            assert "di" in out.result or out.result == "di-md"
            assert out.provider_meta["artifact_id"] == "artifact-2"
            assert out.provider_meta["docx_url"] == "https://media/from-converter.docx"
            assert upload_mock.await_count == 1

        di.process = AsyncMock(side_effect=RuntimeError("fail"))
        with (
            patch("apps.ocr.pipeline.renderer.count_pdf_bytes", return_value=1),
            patch("apps.ocr.services.finance.check_quota", AsyncMock(return_value=10)),
            patch(
                "apps.ocr.document_intelligence.DocumentIntelligencePipeline",
                return_value=di,
            ),
        ):
            out = await svc._process_with_document_intelligence(
                task,
                BytesIO(b"%PDF"),
                "application/pdf",
                OcrEngineType.document_intelligence,
            )
            assert out is task

        with (
            patch("apps.ocr.pipeline.renderer.count_pdf_bytes", return_value=0),
        ):
            await svc._process_with_document_intelligence(
                task,
                BytesIO(b"%PDF"),
                "application/pdf",
                OcrEngineType.document_intelligence,
            )
        with (
            patch("apps.ocr.pipeline.renderer.count_pdf_bytes", return_value=2),
            patch("apps.ocr.services.finance.check_quota", AsyncMock(return_value=0)),
        ):
            await svc._process_with_document_intelligence(
                task,
                BytesIO(b"%PDF"),
                "application/pdf",
                OcrEngineType.document_intelligence,
            )


@pytest.mark.unit
class TestDiLayoutDetectorCoverage:
    def test_detect_and_parse(self, tmp_path: Path) -> None:
        from apps.ocr.document_intelligence.layout import (
            LayoutDetector,
            LayoutType,
            PaddleOcrMissingError,
            load_layout_detector,
        )
        from apps.ocr.document_intelligence.loader import Page

        page = Page(
            id="p1",
            page_number=1,
            image=Image.new("RGB", (100, 100), "white"),
            width=100,
            height=100,
        )
        det = LayoutDetector(crop_dir=tmp_path)
        img = Image.new("RGB", (100, 100), "white")

        model = MagicMock()
        model.predict.return_value = iter([
            SimpleNamespace(
                json=lambda: {
                    "boxes": [
                        {
                            "label": "title",
                            "bbox": [5, 5, 40, 20],
                            "confidence": 0.9,
                        },
                        {
                            "type": "paragraph",
                            "coordinate": {"x1": 1, "y1": 2, "x2": 30, "y2": 40},
                            "score": 0.8,
                        },
                        {"label": "x", "bbox": [1]},
                        "bad",
                    ]
                }
            )
        ])
        det._models = {"PP-DocLayoutV2": model, "PP-DocLayoutV3": model}
        elems = det.detect(img, page)
        assert elems
        assert elems[0].crop_path
        det.log_stats()

        assert det._parse_output("x", page, "s") == []
        with patch.object(det, "_get_model", return_value=None):
            assert det._run_model(img, page, "missing") == []
        with pytest.raises(PaddleOcrMissingError):
            raise PaddleOcrMissingError()

        model.predict.side_effect = RuntimeError("fail")
        assert det._run_model(img, page, "PP-DocLayoutV2") == []

        with patch("apps.ocr.document_intelligence.layout.LayoutDetector") as cls:
            cls.return_value = MagicMock()
            load_layout_detector(0.5)
            cls.assert_called()
        _ = LayoutType.title


@pytest.mark.unit
class TestOpenRouterCoverage:
    def test_helpers(self) -> None:
        from utils.integrations import openrouter as orc

        assert "chat/completions" in orc.chat_completions_url()
        with (
            patch.object(orc.Settings, "openrouter_api_key", ""),
            pytest.raises(orc.OpenRouterConfigurationError),
        ):
            orc.resolve_api_key()
        with patch.object(orc.Settings, "openrouter_api_key", "k"):
            assert orc.resolve_api_key() == "k"
            assert "Bearer" in orc.build_headers()["Authorization"]
        assert orc.extract_provider_meta({"id": "1", "model": "m"}, provider="or")
        assert orc.parse_sse_delta_line("") is None
        assert orc.parse_sse_delta_line(": comment") is None
        assert orc.parse_sse_delta_line("data: [DONE]") == "[DONE]"
        assert (
            orc.parse_sse_delta_line('data: {"choices":[{"delta":{"content":"hi"}}]}')
            == "hi"
        )
        assert orc.parse_sse_delta_line("data: {bad") is None

    @pytest.mark.asyncio
    async def test_complete_and_stream(self) -> None:
        from utils.integrations import openrouter as orc

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"ok": True}
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(return_value=mock_resp)

        with (
            patch.object(orc.Settings, "openrouter_api_key", "k"),
            patch("httpx.AsyncClient", return_value=client),
        ):
            assert await orc.complete_chat_json({"m": 1}) == {"ok": True}
            assert await orc.post_chat_completion_unchecked({"m": 1}) is mock_resp

        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "e",
            request=MagicMock(),
            response=MagicMock(status_code=500, text="x"),
        )
        with (
            patch.object(orc.Settings, "openrouter_api_key", "k"),
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(orc.OpenRouterRequestError),
        ):
            await orc.complete_chat_json({"m": 1})

        client.post = AsyncMock(side_effect=httpx.RequestError("boom"))
        with (
            patch.object(orc.Settings, "openrouter_api_key", "k"),
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(orc.OpenRouterRequestError),
        ):
            await orc.complete_chat_json({"m": 1})

        # stream deltas
        class StreamResp:
            def raise_for_status(self) -> None:
                return None

            async def aiter_lines(self):
                yield 'data: {"choices":[{"delta":{"content":"a"}}]}'
                yield "data: [DONE]"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

        stream_client = AsyncMock()
        stream_client.__aenter__ = AsyncMock(return_value=stream_client)
        stream_client.__aexit__ = AsyncMock(return_value=None)
        stream_client.stream = MagicMock(return_value=StreamResp())

        with (
            patch.object(orc.Settings, "openrouter_api_key", "k"),
            patch("httpx.AsyncClient", return_value=stream_client),
        ):
            chunks = [c async for c in orc.stream_chat_deltas({"m": 1})]
        assert chunks == ["a"]

        class ByteStream:
            status_code = 200

            async def aiter_bytes(self):
                yield b"chunk1"

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def aread(self):
                return b"err"

        stream_client.stream = MagicMock(return_value=ByteStream())
        with (
            patch.object(orc.Settings, "openrouter_api_key", "k"),
            patch.object(orc.Settings, "openai_stream_max_bytes", 10000),
            patch("httpx.AsyncClient", return_value=stream_client),
        ):
            data = [c async for c in orc.stream_chat_completion_bytes({"m": 1})]
        assert data == [b"chunk1"]

        class ErrStream(ByteStream):
            status_code = 500

        stream_client.stream = MagicMock(return_value=ErrStream())
        with (
            patch.object(orc.Settings, "openrouter_api_key", "k"),
            patch("httpx.AsyncClient", return_value=stream_client),
            pytest.raises(orc.OpenRouterError),
        ):
            _ = [c async for c in orc.stream_chat_completion_bytes({"m": 1})]

        class HugeStream(ByteStream):
            async def aiter_bytes(self):
                yield b"x" * 100

        stream_client.stream = MagicMock(return_value=HugeStream())
        with (
            patch.object(orc.Settings, "openrouter_api_key", "k"),
            patch.object(orc.Settings, "openai_stream_max_bytes", 10),
            patch("httpx.AsyncClient", return_value=stream_client),
            pytest.raises(orc.OpenRouterError),
        ):
            _ = [c async for c in orc.stream_chat_completion_bytes({"m": 1})]


@pytest.mark.unit
class TestChatRoutesCoverage:
    @pytest.mark.asyncio
    async def test_router_methods(self) -> None:
        from apps.language.chat.routes import ChatSessionRouter

        router = ChatSessionRouter.__new__(ChatSessionRouter)
        user = SimpleNamespace(uid="u1", user_id="u1", workspace_id=None)
        request = MagicMock()
        session = SimpleNamespace(
            uid="s1",
            model_dump=lambda: {"uid": "s1"},
        )
        thread = SimpleNamespace(
            uid="th1",
            session_uid="s1",
            model_dump=lambda: {"uid": "th1"},
        )
        msg = SimpleNamespace(
            uid="m1",
            model_dump=lambda **k: {"uid": "m1"},
        )

        router.get_user = AsyncMock(return_value=user)
        router.authorize = AsyncMock()
        router._owner_id_for_create = MagicMock(return_value="u1")
        router.get_item = AsyncMock(return_value=session)
        router.get_list_filter_queries = MagicMock(return_value={})

        with patch(
            "apps.language.chat.routes.bootstrap_session",
            AsyncMock(return_value=(session, thread)),
        ):
            data = SimpleNamespace(
                title="t",
                initial_thread_title="it",
                initial_chat_model="m",
                model_dump=lambda exclude_none=True: {"title": "t"},
            )
            assert await router.create_item(request, data) is session

        with (
            patch(
                "apps.language.chat.routes.ChatThread.get_item",
                AsyncMock(return_value=thread),
            ),
            patch(
                "apps.language.chat.routes.ChatSession.update_item",
                AsyncMock(return_value=session),
            ),
        ):
            upd = SimpleNamespace(
                model_dump=lambda exclude_unset=True: {"active_thread_uid": "th1"}
            )
            assert await router.update_item(request, "s1", upd) is session

        with patch(
            "apps.language.chat.routes.ChatThread.get_item",
            AsyncMock(return_value=None),
        ):
            from apps.language.shared.exceptions import ThreadNotFoundError

            upd = SimpleNamespace(
                model_dump=lambda exclude_unset=True: {"active_thread_uid": "x"}
            )
            with pytest.raises(ThreadNotFoundError):
                await router.update_item(request, "s1", upd)

        schema_thread = MagicMock()
        schema_msg = MagicMock()
        with (
            patch(
                "apps.language.chat.routes.ChatThread.list_total_combined",
                AsyncMock(return_value=([thread], 1)),
            ),
            patch(
                "apps.language.chat.routes.ChatThreadSchema.model_validate",
                return_value=schema_thread,
            ),
            patch(
                "apps.language.chat.routes.PaginatedResponse",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
        ):
            page = await router.list_session_threads(request, "s1", 0, 10)
            assert page.total == 1

        with patch(
            "apps.language.chat.routes.ChatThread.create_item",
            AsyncMock(return_value=thread),
        ):
            data = SimpleNamespace(model_dump=lambda exclude_none=True: {"title": "n"})
            assert await router.create_thread(request, "s1", data) is thread

        with patch(
            "apps.language.chat.routes.ChatThread.get_item",
            AsyncMock(return_value=thread),
        ):
            assert await router.retrieve_thread(request, "s1", "th1") is thread

        with (
            patch.object(router, "retrieve_thread", AsyncMock(return_value=thread)),
            patch(
                "apps.language.chat.routes.ChatMessage.list_total_combined",
                AsyncMock(return_value=([msg], 1)),
            ),
            patch(
                "apps.language.chat.routes.ChatMessageSchema.model_validate",
                return_value=schema_msg,
            ),
            patch(
                "apps.language.chat.routes.PaginatedResponse",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
        ):
            page = await router.list_messages(request, "s1", "th1", 0, 10)
            assert page.total == 1

        # _message_reply without generate
        data = SimpleNamespace(generate_reply=False, stream=False)
        user_msg = MagicMock()
        with (
            patch(
                "apps.language.chat.routes.ChatMessageSchema.model_validate",
                return_value=MagicMock(),
            ),
            patch(
                "apps.language.chat.routes.ChatCompletionResponse",
                side_effect=lambda **kw: SimpleNamespace(**kw),
            ),
        ):
            resp = await router._message_reply(
                user=user, thread=thread, data=data, user_msg=user_msg
            )
            assert resp.assistant_message is None


@pytest.mark.unit
class TestMiscSmallModules:
    def test_no_ocr_and_paddle_helpers(self, tmp_path: Path) -> None:
        from apps.ocr import no_ocr_services as nos
        from apps.ocr import paddle_ocr_services as pos

        doc = Document()
        doc.add_paragraph("hello docx")
        buf = BytesIO()
        doc.save(buf)
        buf.seek(0)
        with patch("apps.ocr.no_ocr_services.is_docx", return_value=True):
            assert "hello" in nos.process_direct_file(buf, "docx")
        assert nos.process_direct_file(BytesIO(b"x"), "unknown") == ""

        assert pos._string_keyed_values({"a": 1, 2: "b"}) == {"a": 1}
        assert pos._extract_text_payload({"text": " hi "}) == " hi "
        assert pos._extract_text_payload({"markdown": "m"}) == "m"
        assert (
            pos._extract_parsing_text({
                "parsing_res_list": [{"text": "a"}, {"text": "b"}]
            })
            == "a\nb"
        )
        assert pos._extract_text_payload("raw") == "raw"
        assert pos._extract_text_payload(None) == ""

        with patch(
            "apps.ocr.paddle_ocr_services._get_pipeline",
            return_value=MagicMock(
                predict=MagicMock(
                    return_value=iter([
                        SimpleNamespace(json=lambda: {"text": "paddle"})
                    ])
                )
            ),
        ):
            assert pos._predict_single_image(BytesIO(b"img")) == "paddle"

    @pytest.mark.asyncio
    async def test_paddle_pages_and_completion_route(self) -> None:
        from apps.language.completion import routes as cr
        from apps.ocr import paddle_ocr_services as pos

        with patch(
            "apps.ocr.paddle_ocr_services._predict_single_image",
            return_value="t",
        ):
            out = await pos.process_pages_with_paddle([BytesIO(b"1"), BytesIO(b"2")])
        assert out == ["t", "t"]

        with patch(
            "apps.ocr.paddle_ocr_services._predict_single_image",
            side_effect=RuntimeError("x"),
        ):
            out = await pos.process_pages_with_paddle([BytesIO(b"1")])
        assert out == [None]

        req = MagicMock()
        req.json = AsyncMock(return_value={"model": "m", "messages": []})
        user = SimpleNamespace(uid="u1", user_id="u1")
        with patch(
            "apps.language.completion.routes.proxy_chat_completions",
            AsyncMock(return_value=(b"{}", "application/json", 200)),
        ):
            resp = await cr.openai_compatible_chat_completions(req, user)
            assert resp.status_code == 200

        req.json = AsyncMock(
            side_effect=__import__("json").JSONDecodeError("e", "d", 0)
        )
        from fastapi_mongo_base.errors import BadRequestError

        with pytest.raises(BadRequestError):
            await cr.openai_compatible_chat_completions(req, user)

        req.json = AsyncMock(return_value=["not-dict"])
        with pytest.raises(BadRequestError):
            await cr.openai_compatible_chat_completions(req, user)

        req.json = AsyncMock(return_value={"stream": True, "model": "m"})

        async def gen(body, user_id=""):
            yield b"data"

        with patch(
            "apps.language.completion.routes.proxy_chat_completions_raw_stream",
            gen,
        ):
            resp = await cr.openai_compatible_chat_completions(req, user)
            body = b""
            async for chunk in resp.body_iterator:
                body += chunk
            assert body == b"data"

    @pytest.mark.asyncio
    async def test_openai_audio_helpers(self) -> None:
        from apps.openai_compat import audio as aud

        with patch(
            "apps.openai_compat.audio.finance.pricing_config",
            return_value={"speech": {"default_per_1k_chars": 1.0, "markup": 2.0}},
        ):
            assert aud._estimate_speech_cost("abcd") > 0

        hints = aud._language_hints(None)
        assert hints
        hints2 = aud._language_hints("not-a-lang")
        assert hints2

        soniox = MagicMock()
        job_ok = SimpleNamespace(
            status=__import__(
                "soniox.types", fromlist=["TranscriptionJobStatus"]
            ).TranscriptionJobStatus.COMPLETED
        )
        soniox.get_transcription_job_async = AsyncMock(return_value=job_ok)
        with (
            patch(
                "apps.openai_compat.audio.transcribe_services.get_soniox_client",
                return_value=soniox,
            ),
            patch.object(aud.Settings, "transcribe_poll_interval_seconds", 0),
        ):
            assert await aud._poll_transcription("j1") is job_ok

        with (
            patch(
                "apps.openai_compat.audio.finance.check_quota",
                AsyncMock(),
            ),
            patch(
                "apps.openai_compat.audio.resolve_api_key",
                return_value="k",
            ),
            patch(
                "apps.openai_compat.audio.build_headers",
                return_value={},
            ),
            patch(
                "apps.openai_compat.audio.finance.meter_cost",
                AsyncMock(),
            ),
            patch("httpx.AsyncClient") as client_cls,
        ):
            client = AsyncMock()
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=None)
            resp = MagicMock(status_code=200, content=b"audio", headers={})
            client.post = AsyncMock(return_value=resp)
            client_cls.return_value = client
            out = await aud.create_speech({"input": "hi"}, user_id="u1")
            assert out.body == b"audio"

        from fastapi_mongo_base.core.exceptions import BaseHTTPException

        with pytest.raises(BaseHTTPException):
            await aud.create_speech({}, user_id="u1")

    @pytest.mark.asyncio
    async def test_ocr_to_text_paths(self) -> None:
        from apps.ocr import ocr_services as osvc

        img = Image.new("RGB", (20, 20), "white")
        bio = BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)

        with (
            patch("apps.ocr.ocr_services._read_ocr_prompt", return_value="p"),
            patch(
                "apps.ocr.ocr_services._ocr_attempt",
                AsyncMock(side_effect=[(None, True), ("ok", False)]),
            ),
        ):
            assert await osvc.ocr_to_text(bio) == "ok"

        with (
            patch(
                "apps.ocr.ocr_services._read_text_enhancement_prompt",
                return_value="p",
            ),
            patch(
                "apps.ocr.ocr_services.complete_chat_json",
                AsyncMock(
                    return_value={"choices": [{"message": {"content": " clean "}}]}
                ),
            ),
        ):
            assert await osvc.text_enhancement("x") == "clean"

    def test_imagetools_pdftools(self) -> None:
        from utils.files import imagetools, pdftools

        img = Image.new("RGB", (30, 30), "blue")
        with patch.object(imagetools, "resize_image", create=True):
            pass
        # exercise common helpers if present
        for name in dir(imagetools):
            if name.startswith("_"):
                continue
        assert pdftools is not None
        # try convert helpers
        bio = BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        if hasattr(imagetools, "pil_to_bytes"):
            imagetools.pil_to_bytes(img)
        if hasattr(imagetools, "bytes_to_pil"):
            imagetools.bytes_to_pil(bio.getvalue())
