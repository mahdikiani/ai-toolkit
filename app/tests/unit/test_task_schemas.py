"""Unit tests for shared task create field mixin."""

import pytest
from fastapi_mongo_base.tasks import TaskCreateFieldsMixin

from apps.youtube.schemas import YoutubeTranscriptTaskSchemaCreate


@pytest.mark.unit
class TestTaskCreateFieldsMixin:
    """Tests for TaskCreateFieldsMixin."""

    def test_defaults_are_optional(self) -> None:
        """Task create fields should default to None."""
        fields = TaskCreateFieldsMixin()
        assert fields.user_id is None
        assert fields.webhook_url is None
        assert fields.meta_data is None

    def test_youtube_create_accepts_webhook_and_meta_data(self) -> None:
        """YouTube create schema should accept shared task fields."""
        task = YoutubeTranscriptTaskSchemaCreate(
            video_id="dQw4w9WgXcQ",
            webhook_url="https://example.com/hook",
            meta_data={"chat_id": 123},
        )
        assert task.webhook_url == "https://example.com/hook"
        assert task.meta_data == {"chat_id": 123}

    @pytest.mark.parametrize(
        ("video_id", "expected"),
        [
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            (
                "https://youtu.be/URKml8lgw8Y?si=VZPsMb2hNFOAxvJe",
                "URKml8lgw8Y",
            ),
            (
                "https://www.youtube.com/watch?v=3QU-_PSbKlo&t=10s",
                "3QU-_PSbKlo",
            ),
            (
                "http://youtube.com/watch?v=3QU-_PSbKlo",
                "3QU-_PSbKlo",
            ),
            (
                "https://www.youtube.com/shorts/abc123XYZ-_",
                "abc123XYZ-_",
            ),
        ],
    )
    def test_youtube_create_normalizes_video_urls(
        self,
        video_id: str,
        expected: str,
    ) -> None:
        """YouTube create schema should normalize supported URLs to video ids."""
        task = YoutubeTranscriptTaskSchemaCreate(video_id=video_id)
        assert task.video_id == expected
