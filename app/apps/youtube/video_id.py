"""YouTube video ID extraction from ids and common URL formats."""

from urllib.parse import parse_qs, urlparse


class YouTubeVideoIdRequiredError(ValueError):
    """Raised when video_id is empty."""

    def __init__(self) -> None:
        """Initialize with the default validation message."""
        super().__init__("video_id is required")


class InvalidYouTubeURLError(ValueError):
    """Raised when a YouTube URL cannot be parsed."""

    def __init__(self) -> None:
        """Initialize with the default validation message."""
        super().__init__("Invalid YouTube URL")


class YouTubeVideoIdTypeError(TypeError):
    """Raised when video_id is not a string."""

    def __init__(self) -> None:
        """Initialize with the default validation message."""
        super().__init__("video_id must be a string")


def _youtube_host(netloc: str) -> str:
    return netloc.lower().rsplit("@", 1)[-1].rsplit(":", 1)[0]


def _is_youtube_host(netloc: str) -> bool:
    host = _youtube_host(netloc)
    return (
        host == "youtu.be" or host.endswith(".youtu.be") or host.endswith("youtube.com")
    )


def parse_youtube_video_id(value: str) -> str:
    """Extract a YouTube video id from a raw id or common YouTube URL."""
    candidate = value.strip()
    if not candidate:
        raise YouTubeVideoIdRequiredError

    parsed = urlparse(candidate)
    if not parsed.netloc:
        return candidate

    if parsed.scheme not in ("http", "https") or not _is_youtube_host(parsed.netloc):
        raise InvalidYouTubeURLError

    host = _youtube_host(parsed.netloc)
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif query_video := parse_qs(parsed.query).get("v"):
        video_id = query_video[0]
    elif "/shorts/" in parsed.path:
        video_id = parsed.path.split("/shorts/", 1)[1].split("/", 1)[0]
    elif "/embed/" in parsed.path:
        video_id = parsed.path.split("/embed/", 1)[1].split("/", 1)[0]
    elif parsed.path.startswith("/v/"):
        video_id = parsed.path.removeprefix("/v/").split("/", 1)[0]
    else:
        raise InvalidYouTubeURLError

    if not video_id:
        raise InvalidYouTubeURLError
    return video_id
