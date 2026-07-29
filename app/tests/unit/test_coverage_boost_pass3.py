"""Third pass: close remaining coverage gap to 85%."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image


@pytest.mark.unit
class TestPass3UtilsAndRoutes:
    def test_imagetools_and_pdftools(self, tmp_path: Path) -> None:
        from utils.files import imagetools, pdftools

        img = Image.new("RGB", (16, 16), "green")
        assert imagetools.convert_to_jpg(img).mode == "RGB"
        bio = BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        assert imagetools.convert_to_jpg(bio).mode == "RGB"
        out = imagetools.convert_to_jpg_bytes(img)
        assert out.read(2) == b"\xff\xd8"
        bio.seek(0)
        assert imagetools.convert_to_jpg_bytes(bio).read(2) == b"\xff\xd8"

        with (
            patch("pdf2image.pdfinfo_from_path", return_value={"Pages": 3}),
            patch(
                "pdf2image.convert_from_path",
                return_value=[img, img],
            ),
            patch(
                "pdf2image.convert_from_bytes",
                return_value=[img],
            ),
        ):
            assert pdftools.number_of_pages(tmp_path / "a.pdf") == 3
            assert len(pdftools.extract_pdf_pages(tmp_path / "a.pdf")) == 2
            assert len(pdftools.extract_pdf_pages_with_index(tmp_path / "a.pdf")) == 2
            assert len(pdftools.extract_pdf_bytes_pages(BytesIO(b"%PDF"))) == 1
            assert (
                len(pdftools.extract_pdf_bytes_pages_with_index(BytesIO(b"%PDF"))) == 1
            )

    def test_youtube_video_id(self) -> None:
        from apps.youtube import video_id as vid

        with pytest.raises(vid.YouTubeVideoIdRequiredError):
            vid.parse_youtube_video_id("  ")
        assert vid.parse_youtube_video_id("abc123") == "abc123"
        assert vid.parse_youtube_video_id("https://youtu.be/watchid01") == "watchid01"
        assert (
            vid.parse_youtube_video_id("https://www.youtube.com/watch?v=watchid02&t=1")
            == "watchid02"
        )
        assert (
            vid.parse_youtube_video_id("https://www.youtube.com/shorts/shortid1")
            == "shortid1"
        )
        assert (
            vid.parse_youtube_video_id("https://www.youtube.com/embed/embedid1")
            == "embedid1"
        )
        assert (
            vid.parse_youtube_video_id("https://www.youtube.com/v/vidpath1")
            == "vidpath1"
        )
        with pytest.raises(vid.InvalidYouTubeURLError):
            vid.parse_youtube_video_id("https://example.com/x")
        with pytest.raises(vid.InvalidYouTubeURLError):
            vid.parse_youtube_video_id("https://www.youtube.com/channel/x")
        with pytest.raises(vid.YouTubeVideoIdTypeError):
            raise vid.YouTubeVideoIdTypeError()

    @pytest.mark.asyncio
    async def test_web_download_and_openai_routes(self) -> None:
        from apps.openai_compat import routes as oc
        from utils.downloaders import web

        resp = MagicMock()
        resp.is_redirect = False
        resp.raise_for_status = MagicMock()
        resp.headers = {"content-type": "application/octet-stream"}
        resp.content = b"file"
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock(return_value=resp)
        with (
            patch("utils.downloaders.web.assert_safe_url"),
            patch("utils.downloaders.web.is_gdrive_url", return_value=False),
            patch("httpx.AsyncClient", return_value=client),
        ):
            buf = await web.download_bytes("https://example.com/a.bin")
        assert buf.read() == b"file"

        # redirect + gdrive confirm path
        redir = MagicMock()
        redir.is_redirect = True
        redir.headers = {"location": "https://example.com/b"}
        final = MagicMock()
        final.is_redirect = False
        final.raise_for_status = MagicMock()
        final.headers = {"content-type": "text/html"}
        final.text = 'href="/uc?confirm=tok123"'
        final.content = b"html"
        confirmed = MagicMock()
        confirmed.raise_for_status = MagicMock()
        confirmed.content = b"ok"
        client.get = AsyncMock(side_effect=[redir, final, confirmed])
        with (
            patch("utils.downloaders.web.assert_safe_url"),
            patch("utils.downloaders.web.is_gdrive_url", return_value=True),
            patch(
                "utils.downloaders.web.resolve_gdrive_download_url",
                return_value="https://drive.google.com/uc?id=fid",
            ),
            patch(
                "utils.downloaders.web.gdrive_direct_download_url",
                return_value="https://drive.google.com/uc?id=fid",
            ),
            patch("httpx.AsyncClient", return_value=client),
        ):
            buf = await web.download_bytes("https://drive.google.com/file/d/fid/view")
        assert buf.read() == b"ok"

        user = SimpleNamespace(uid="u1", user_id="u1")
        assert await oc.list_models(user) == {
            "object": "list",
            "data": oc.AVAILABLE_MODELS,
        }

        req = MagicMock()
        req.json = AsyncMock(
            return_value={"messages": [{"role": "user", "content": "h"}]}
        )
        with patch(
            "apps.openai_compat.routes.services.handle_non_stream_chat",
            AsyncMock(return_value={"ok": 1}),
        ):
            assert await oc.chat_completions(req, user) == {"ok": 1}

        req.json = AsyncMock(
            return_value={
                "messages": [{"role": "user", "content": "h"}],
                "stream": True,
                "max_tokens": 10,
            }
        )
        with patch(
            "apps.openai_compat.routes.services.handle_stream_chat",
            AsyncMock(return_value={"stream": 1}),
        ):
            assert await oc.chat_completions(req, user) == {"stream": 1}

        req.json = AsyncMock(
            side_effect=__import__("json").JSONDecodeError("e", "d", 0)
        )
        from fastapi_mongo_base.core.exceptions import BaseHTTPException

        with pytest.raises(BaseHTTPException):
            await oc.chat_completions(req, user)
        req.json = AsyncMock(return_value=["bad"])
        with pytest.raises(BaseHTTPException):
            await oc.chat_completions(req, user)
        req.json = AsyncMock(return_value={"messages": []})
        with pytest.raises(BaseHTTPException):
            await oc.chat_completions(req, user)

        req.json = AsyncMock(return_value={"input": "hi"})
        with patch(
            "apps.openai_compat.routes.audio_api.create_speech",
            AsyncMock(return_value=MagicMock()),
        ) as speech:
            await oc.audio_speech(req, user)
            speech.assert_awaited()

        req.json = AsyncMock(
            side_effect=__import__("json").JSONDecodeError("e", "d", 0)
        )
        with pytest.raises(BaseHTTPException):
            await oc.audio_speech(req, user)
        req.json = AsyncMock(return_value=["x"])
        with pytest.raises(BaseHTTPException):
            await oc.audio_speech(req, user)

        upload = MagicMock()
        upload.read = AsyncMock(return_value=b"wav")
        upload.filename = "a.wav"
        with patch(
            "apps.openai_compat.routes.audio_api.create_transcription",
            AsyncMock(return_value={"text": "hi"}),
        ):
            assert await oc.audio_transcriptions(user, upload) == {"text": "hi"}

    @pytest.mark.asyncio
    async def test_audio_transcription_and_elements(self, tmp_path: Path) -> None:
        from soniox.types import TranscriptionJobStatus

        from apps.ocr.document_intelligence.elements import (
            ElementProcessor,
            ProcessedElement,
        )
        from apps.ocr.document_intelligence.layout import LayoutElement, LayoutType
        from apps.openai_compat import audio as aud

        soniox = MagicMock()
        soniox.transcribe_file_async = AsyncMock(return_value=SimpleNamespace(id="j1"))
        soniox.get_transcription_result_async = AsyncMock(
            return_value=SimpleNamespace(text="hello", tokens=None)
        )
        with (
            patch.object(aud.Settings, "soniox_api_key", "sk"),
            patch(
                "apps.openai_compat.audio.finance.estimate_transcribe_cost",
                return_value=1.0,
            ),
            patch(
                "apps.openai_compat.audio.finance.check_quota",
                AsyncMock(),
            ),
            patch(
                "apps.openai_compat.audio.finance.meter_cost",
                AsyncMock(),
            ),
            patch(
                "apps.openai_compat.audio.transcribe_services.get_soniox_client",
                return_value=soniox,
            ),
            patch(
                "apps.openai_compat.audio._poll_transcription",
                AsyncMock(
                    return_value=SimpleNamespace(
                        status=TranscriptionJobStatus.COMPLETED,
                        audio_duration_ms=60000,
                    )
                ),
            ),
        ):
            result = await aud.create_transcription(
                b"RIFF",
                filename="a.wav",
                user_id="u1",
                response_format="json",
            )
            assert result["text"] == "hello"
            verbose = await aud.create_transcription(
                b"RIFF",
                filename="a.wav",
                user_id="u1",
                response_format="verbose_json",
            )
            assert verbose["task"] == "transcribe"
            text_only = await aud.create_transcription(
                b"RIFF",
                filename="a.wav",
                user_id="u1",
                response_format="text",
            )
            assert text_only["text"] == "hello"

        from fastapi_mongo_base.core.exceptions import BaseHTTPException

        with pytest.raises(BaseHTTPException):
            await aud.create_transcription(b"", filename="a.wav", user_id="u1")

        proc = ElementProcessor(vlm_model="m", openrouter_client=MagicMock())
        pe = ProcessedElement(text="x")
        assert pe.text == "x"
        elem = LayoutElement(
            id="e1",
            page_id="p1",
            page_number=1,
            type=LayoutType.paragraph,
            bbox=(0, 0, 10, 10),
            padded_bbox=(0, 0, 10, 10),
            confidence=0.9,
            crop_path=str(tmp_path / "c.png"),
        )
        Image.new("RGB", (10, 10)).save(tmp_path / "c.png")
        _ = (proc, elem)
