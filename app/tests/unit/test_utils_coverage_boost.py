# Targeted unit tests for low-coverage utility modules.

from __future__ import annotations

import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from utils.downloaders import gdrive
from utils.files import archive_utils, b64tools
from utils.integrations import media


@pytest.mark.unit
class TestB64Tools:
    def test_b64_image(self) -> None:
        img = Image.new("RGB", (8, 8), color="red")
        result = b64tools.b64_file(img)
        assert result.startswith("data:image/jpeg;base64,")

    def test_b64_bytes(self) -> None:
        data = BytesIO(b"%PDF-1.4 sample")
        result = b64tools.b64_file(data)
        assert result.startswith("data:")

    def test_b64_file_path(self, tmp_path: Path) -> None:
        sample = tmp_path / "note.txt"
        sample.write_text("hello", encoding="utf-8")
        result = b64tools.b64_file(sample)
        assert "base64," in result


@pytest.mark.unit
class TestGdriveHelpers:
    def test_extract_file_id_from_share_link(self) -> None:
        url = "https://drive.google.com/file/d/abc123XYZ/view"
        assert gdrive.extract_gdrive_file_id(url) == "abc123XYZ"

    def test_extract_file_id_from_docs_link(self) -> None:
        url = "https://docs.google.com/document/d/doc-id/edit"
        assert gdrive.extract_gdrive_file_id(url) == "doc-id"

    def test_is_gdrive_url(self) -> None:
        assert gdrive.is_gdrive_url("https://drive.google.com/uc?export=download&id=x")

    def test_resolve_and_confirm_token(self) -> None:
        url = "https://drive.google.com/file/d/file123/view"
        resolved = gdrive.resolve_gdrive_download_url(url)
        assert "file123" in resolved
        token = gdrive.parse_large_file_confirm_token('href="/uc?confirm=abc123"')
        assert token == "abc123"


@pytest.mark.unit
class TestMediaClient:
    async def test_upload_file(self) -> None:
        mock_upload = MagicMock()
        mock_upload.json.return_value = {"uid": "f1", "url": "https://media/x"}
        mock_upload.raise_for_status = MagicMock()
        mock_patch = MagicMock()
        mock_patch.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_upload)
        mock_client.patch = AsyncMock(return_value=mock_patch)

        with patch(
            "utils.integrations.media.get_media_client",
        ) as get_client:
            get_client.return_value.__aenter__.return_value = mock_client
            get_client.return_value.__aexit__.return_value = None
            url = await media.upload_file(BytesIO(b"data"))

        assert url == "https://media/x"


@pytest.mark.unit
class TestArchiveUtilsAsync:
    async def test_run_directory_files(self, tmp_path: Path) -> None:
        sample = tmp_path / "a.txt"
        sample.write_text("x", encoding="utf-8")

        results = await archive_utils.run_directory_files(
            tmp_path,
            lambda path: path.name,
        )
        assert "a.txt" in results

    async def test_process_directory_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        (src / "a.txt").write_text("content", encoding="utf-8")

        written = await archive_utils.process_directory_files(
            src,
            out,
            lambda path: f"processed:{path.name}",
        )
        assert written
        assert written[0].read_text(encoding="utf-8") == "processed:a.txt"

    def test_extract_tar(self) -> None:
        buf = BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            data = tarfile.TarInfo(name="safe.txt")
            payload = b"tar-data"
            data.size = len(payload)
            tar.addfile(data, BytesIO(payload))
        buf.seek(0)
        result = archive_utils.extract_archive(buf, "application/x-tar")
        assert result is not None
        _, paths = result
        assert paths[0].read_bytes() == b"tar-data"
