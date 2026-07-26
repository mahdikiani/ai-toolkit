# Tests for SSRF guards, zip-slip, and Soniox webhook auth.

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from apps.transcribe.webhook_auth import (
    WebhookAuthError,
    append_webhook_auth,
    build_webhook_token,
    verify_webhook_request,
)
from utils.downloaders.url_safety import UnsafeUrlError, assert_safe_url
from utils.files.archive_utils import _safe_member_path, extract_archive


class TestUrlSafety:
    def test_allows_https_public_host(self) -> None:
        with patch(
            "utils.downloaders.url_safety.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("93.184.216.34", 0))],
        ):
            assert_safe_url("https://example.com/file.pdf")

    def test_blocks_localhost(self) -> None:
        with pytest.raises(UnsafeUrlError):
            assert_safe_url("http://localhost/secret")

    def test_blocks_private_ip_literal(self) -> None:
        with pytest.raises(UnsafeUrlError):
            assert_safe_url("http://127.0.0.1/x")
        with pytest.raises(UnsafeUrlError):
            assert_safe_url("http://192.168.1.10/x")
        with pytest.raises(UnsafeUrlError):
            assert_safe_url("http://169.254.169.254/latest/meta-data")

    def test_blocks_dns_to_private(self) -> None:
        with (
            patch(
                "utils.downloaders.url_safety.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("10.0.0.5", 0))],
            ),
            pytest.raises(UnsafeUrlError),
        ):
            assert_safe_url("https://evil.example/file")

    def test_allows_data_urls(self) -> None:
        assert_safe_url("data:text/plain;base64,YQ==")


class TestZipSlip:
    def test_rejects_traversal_member(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            _safe_member_path(tmp_path, "../etc/passwd")

    def test_accepts_nested_safe_member(self, tmp_path: Path) -> None:
        path = _safe_member_path(tmp_path, "docs/a.txt")
        assert path.parent == (tmp_path / "docs").resolve()

    def test_extract_zip_blocks_slip(self, tmp_path: Path) -> None:
        import zipfile

        buf = BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../evil.txt", "nope")
        buf.seek(0)
        result = extract_archive(buf, "application/zip")
        assert result is None


class TestSonioxWebhookAuth:
    def test_fail_closed_without_secret(self) -> None:
        with patch("apps.transcribe.webhook_auth.Settings") as settings:
            settings.soniox_webhook_secret = None
            with pytest.raises(WebhookAuthError) as exc:
                verify_webhook_request(uid="abc", token="x")
            assert exc.value.status_code == 503

    def test_accepts_valid_token(self) -> None:
        with patch("apps.transcribe.webhook_auth.Settings") as settings:
            settings.soniox_webhook_secret = "test-secret"
            token = build_webhook_token("task-uid")
            verify_webhook_request(uid="task-uid", token=token)
            url = append_webhook_auth(
                "https://example.com/api/ai/v1/transcribes/task-uid/webhook",
                "task-uid",
            )
            assert "token=" in url

    def test_rejects_bad_token(self) -> None:
        with patch("apps.transcribe.webhook_auth.Settings") as settings:
            settings.soniox_webhook_secret = "test-secret"
            with pytest.raises(WebhookAuthError):
                verify_webhook_request(uid="task-uid", token="deadbeef")
