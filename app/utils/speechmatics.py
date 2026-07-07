"""Speechmatics API client and transcription utilities."""

import json
import os
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self

import httpx
from fastapi_mongo_base.utils import basic
from pydantic import BaseModel
from singleton import Singleton


class Language(StrEnum):
    """Enumeration of supported languages for speech transcription."""

    English = "English"
    Persian = "Persian"
    Arabic = "Arabic"
    Turkish = "Turkish"
    French = "French"
    Spanish = "Spanish"
    German = "German"
    Italian = "Italian"
    Portuguese = "Portuguese"
    Dutch = "Dutch"
    Russian = "Russian"
    Polish = "Polish"
    Romanian = "Romanian"
    Bulgarian = "Bulgarian"
    Hungarian = "Hungarian"
    Czech = "Czech"
    Greek = "Greek"
    Hebrew = "Hebrew"
    Japanese = "Japanese"
    Korean = "Korean"
    Mandarin = "Mandarin"
    # Chinese = "Chinese"
    Vietnamese = "Vietnamese"
    Indonesian = "Indonesian"

    @classmethod
    def has_value(cls, value: str) -> bool:
        """
        Check if a value exists in the Language enum.

        Args:
            value: String value to check.

        Returns:
            True if the value is a valid language.
        """
        return value in cls._value2member_map_

    @property
    def _info(self) -> dict:
        return {
            Language.English: {
                "fa": "انگلیسی",
                "en": "English",
                "abbreviation": "en",
            },
            Language.Persian: {
                "fa": "فارسی",
                "en": "Persian",
                "abbreviation": "fa",
            },
            Language.Arabic: {
                "fa": "عربی",
                "en": "Arabic",
                "abbreviation": "ar",
            },
            Language.Turkish: {
                "fa": "ترکی",
                "en": "Turkish",
                "abbreviation": "tr",
            },
            Language.French: {
                "fa": "فرانسه",
                "en": "French",
                "abbreviation": "fr",
            },
            Language.Spanish: {
                "fa": "اسپانیایی",
                "en": "Spanish",
                "abbreviation": "es",
            },
            Language.German: {
                "fa": "آلمانی",
                "en": "German",
                "abbreviation": "de",
            },
            Language.Italian: {
                "fa": "ایتالیایی",
                "en": "Italian",
                "abbreviation": "it",
            },
            Language.Portuguese: {
                "fa": "پرتغالی",
                "en": "Portuguese",
                "abbreviation": "pt",
            },
            Language.Dutch: {
                "fa": "هالندی",
                "en": "Dutch",
                "abbreviation": "nl",
            },
            Language.Russian: {
                "fa": "روسی",
                "en": "Russian",
                "abbreviation": "ru",
            },
            Language.Polish: {
                "fa": "لهستانی",
                "en": "Polish",
                "abbreviation": "pl",
            },
            Language.Romanian: {
                "fa": "رومانیایی",
                "en": "Romanian",
                "abbreviation": "ro",
            },
            Language.Bulgarian: {
                "fa": "بلغاری",
                "en": "Bulgarian",
                "abbreviation": "bg",
            },
            Language.Hungarian: {
                "fa": "مجارستانی",
                "en": "Hungarian",
                "abbreviation": "hu",
            },
            Language.Czech: {
                "fa": "چک",
                "en": "Czech",
                "abbreviation": "cs",
            },
            Language.Greek: {
                "fa": "یونانی",
                "en": "Greek",
                "abbreviation": "el",
            },
            Language.Hebrew: {
                "fa": "عبری",
                "en": "Hebrew",
                "abbreviation": "he",
            },
            Language.Japanese: {
                "fa": "ژاپنی",
                "en": "Japanese",
                "abbreviation": "ja",
            },
            Language.Korean: {
                "fa": "کره ای",
                "en": "Korean",
                "abbreviation": "ko",
            },
            # Language.Chinese: {
            #     "fa": "چینی",
            #     "en": "Chinese",
            #     "abbreviation": "zh",
            # },
            Language.Mandarin: {
                "fa": "چینی ماندارین",
                "en": "Mandarin",
                "abbreviation": "cmn",
            },
            Language.Vietnamese: {
                "fa": "ویتنامی",
                "en": "Vietnamese",
                "abbreviation": "vi",
            },
            Language.Indonesian: {
                "fa": "اندونزیایی",
                "en": "Indonesian",
                "abbreviation": "id",
            },
        }[self]

    @property
    def fa(self) -> str:
        """
        Persian name of the language.

        Returns:
            Persian language name.
        """
        return self._info["fa"]

    @property
    def en(self) -> str:
        """
        English name of the language.

        Returns:
            English language name.
        """
        return self._info["en"]

    @property
    def abbreviation(self) -> str:
        """
        Language abbreviation code.

        Returns:
            Language abbreviation (e.g., 'en', 'fa').
        """
        return self._info["abbreviation"]

    def get_dict(self) -> dict:
        """
        Get a dictionary representation of the language with all metadata.

        Returns:
            Dictionary containing fa, en, abbreviation, and value.
        """
        return self._info | {"value": self.value}

    @classmethod
    def get_choices(cls) -> list[dict]:
        """
        Get all languages as a list of dictionary choices.

        Returns:
            List of language dictionaries for use in form choices.
        """
        return [item.get_dict() for item in cls]


class Alternative(BaseModel):
    """Represents a transcription alternative with confidence and content."""

    confidence: float
    content: str
    language: str
    speaker: str | None = None


class TranscriptionResult(BaseModel):
    """Represents a single transcription result with timing and alternatives."""

    alternatives: list[Alternative]
    end_time: float
    start_time: float
    type: str


class LanguageIdentification(BaseModel):
    """Contains language identification results from transcription."""

    predicted_language: str


class LanguagePackInfo(BaseModel):
    """Contains metadata about the language pack used for transcription."""

    adapted: bool
    itn: bool
    language_description: str
    word_delimiter: str
    writing_direction: str


class TranscriptionConfig(BaseModel):
    """Basic transcription configuration with diarization and language."""

    diarization: str | None = None
    language: str


class Metadata(BaseModel):
    """Transcription metadata including creation time and configuration."""

    created_at: datetime
    language_identification: LanguageIdentification
    language_pack_info: LanguagePackInfo
    transcription_config: TranscriptionConfig
    type: str


class Job(BaseModel):
    """Represents a Speechmatics transcription job with basic info."""

    id: str
    created_at: datetime
    data_name: str
    duration: int


class TranscribeWebhookSchema(BaseModel):
    """Schema for transcription webhook response with results and metadata."""

    format: str
    job: Job
    metadata: Metadata
    results: list[TranscriptionResult]

    @property
    def duration(self) -> int:
        """
        Duration of the transcribed media in seconds.

        Returns:
            Duration in seconds.
        """
        return self.job.duration

    @property
    def job_id(self) -> str:
        """
        Unique job identifier.

        Returns:
            Job ID string.
        """
        return self.job.id

    @property
    def language(self) -> str:
        """
        Predicted language from transcription.

        Returns:
            Predicted language string.
        """
        return self.metadata.language_identification.predicted_language


class TranscriptionConfigDetails(BaseModel):
    """Detailed transcription configuration with vocabulary and labels."""

    additional_vocab: list[str] | None = None
    channel_diarization_labels: list[str] | None = None
    language: str


class JobConfig(BaseModel):
    """Configuration for a transcription job including notifications."""

    notification_config: list[dict] | None = None
    transcription_config: TranscriptionConfigDetails
    type: str


class JobStatus(StrEnum):
    """Enumeration of possible transcription job statuses."""

    running = "running"
    done = "done"
    rejected = "rejected"

    @classmethod
    def finishes(cls) -> list[Self]:
        """
        Get the list of terminal job statuses.

        Returns:
            List of finished status values (done, rejected).
        """
        return [cls.done, cls.rejected]  # type: ignore[list-item]

    def is_finished(self) -> bool:
        """
        Check if the job status is terminal.

        Returns:
            True if the job is done or rejected.
        """
        return self in self.finishes()


class JobError(BaseModel):
    """Represents an error that occurred during transcription."""

    message: str
    timestamp: datetime


class JobDetails(BaseModel):
    """Detailed information about a transcription job."""

    id: str
    created_at: datetime
    config: JobConfig
    data_name: str
    duration: int
    status: JobStatus
    errors: list[JobError] | None = None


class Speechmatics(metaclass=Singleton):
    """Singleton client for the Speechmatics ASR API."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize Speechmatics client with API key.

        Args:
            api_key: Speechmatics API key. Defaults to SPEECHMATICS_API_KEY env var.
        """
        self.base_url = "https://asr.api.speechmatics.com/v2"
        self.api_key = api_key or os.getenv("SPEECHMATICS_API_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    @basic.retry_execution(delay=5, attempts=3)
    async def create_transcribe_job(
        self,
        video_url: str,
        webhook_url: str | None = None,
        *,
        secret_token: str | None = None,
        diarization: bool = False,
        language: str = "auto",
        enhanced: bool = True,
        **kwargs: object,
    ) -> str:
        """
        Create a new transcription job on Speechmatics.

        Args:
            video_url: URL of the media file to transcribe.
            webhook_url: Optional webhook URL for job completion notification.
            secret_token: Optional secret token for webhook authentication.
            diarization: Whether to enable speaker diarization.
            language: Language code for transcription. Defaults to 'auto'.
            enhanced: Whether to use enhanced operating point.
            **kwargs: Additional keyword arguments.

        Returns:
            The created job ID.
        """
        conf: dict = {
            "type": "transcription",
            "audio_events_config": {"types": ["laughter", "music", "applause"]},
            "transcription_config": {
                "language": language,
                "diarization": "none" if not diarization else "speaker",
                "operating_point": "enhanced" if enhanced else "standard",
                "enable_entities": True,
                "audio_filtering_config": {"volume_threshold": 0},
            },
            "fetch_data": {"url": video_url},
        }
        # conf = {
        #     "type": "transcription",
        #     "transcription_config": {"language": "auto"},
        #     "fetch_data": {"url": video_url},
        # }
        if webhook_url:
            conf["notification_config"] = [
                {
                    "url": webhook_url,
                    # "contents": ["transcript", "data"],
                    "auth_headers": [f"Authorization: {secret_token}"],
                }
            ]
        files = {"config": (None, json.dumps(conf))}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f"{self.base_url}/jobs/",
                headers=self.headers,
                files=files,
            )
            response.raise_for_status()
            return response.json().get("id")

    @basic.retry_execution(delay=5, attempts=3)
    async def get_transcribe_job(self, job_id: str, **kwargs: object) -> JobDetails:
        """
        Get the details of a transcription job.

        Args:
            job_id: The job ID to retrieve.
            **kwargs: Additional keyword arguments.

        Returns:
            JobDetails object with job status and configuration.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url=f"{self.base_url}/jobs/{job_id}",
                headers=self.headers,
            )
            response.raise_for_status()
            return JobDetails.model_validate(response.json().get("job"))

    @basic.retry_execution(delay=5, attempts=3)
    async def get_transcript(
        self, job_id: str, **kwargs: object
    ) -> TranscribeWebhookSchema:
        """
        Get the full transcript for a completed transcription job.

        Args:
            job_id: The job ID to retrieve transcript for.
            **kwargs: Additional keyword arguments.

        Returns:
            TranscribeWebhookSchema containing the full transcription results.
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url=f"{self.base_url}/jobs/{job_id}/transcript",
                headers=self.headers,
            )
            response.raise_for_status()
            return TranscribeWebhookSchema.model_validate(response.json())


def transcript_to_sequence(
    transcript: list[TranscriptionResult],
) -> tuple[list[list[float]], list[str], list[str]]:
    """
    Convert a Speechmatics transcript to sentence sequences with timings.

    Args:
        transcript: List of TranscriptionResult objects from Speechmatics API.

    Returns:
        Tuple of (timings, sentences, languages) where timings is a list of
        [start, end] pairs, sentences is a list of text strings, and
        languages is a list of detected language codes.
    """
    # This function takes in a Speechmatics transcript object returned by
    # the Speechmatics API, which is a list of TranscriptionResult objects
    # and extracts a pair of lists, timings and sentences, that includes
    # sentences and their corresponding time-ranges.

    sentences = []
    sen = []
    timings = []
    temp_timing = []
    languages = []
    temp_language = []

    for obj in transcript:
        sen.append(obj.alternatives[0].content)
        temp_timing.append([obj.start_time, obj.end_time])
        temp_language.append(obj.alternatives[0].language)

        if obj.type == "punctuation" and obj.alternatives[0].content in [".", "?", "!"]:
            sentences.append(" ".join(sen))
            timings.append([temp_timing[0][0], temp_timing[-1][-1]])
            # Get most frequent language in temp_language as mode
            mode_language = max(set(temp_language), key=temp_language.count)
            languages.append(mode_language)
            sen = []
            temp_timing = []

    return timings, sentences, languages


def generate_srt(timings: list[list[float]], sentence_seq: list[str]) -> str:
    """
    Generate SRT subtitle format from timings and sentences.

    Args:
        timings: List of [start, end] time pairs in seconds.
        sentence_seq: List of sentence strings.

    Returns:
        SRT-formatted subtitle string.
    """

    # Function to convert time in seconds to SRT format
    def convert_time(seconds: float) -> str:
        time_delta = timedelta(seconds=seconds)
        hours, remainder = divmod(time_delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = time_delta.microseconds // 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    srt_lines: list[str] = []
    for index, (time_range, sentence) in enumerate(
        zip(timings, sentence_seq, strict=True), start=1
    ):
        start, end = time_range
        srt_lines.append(f"{index}")
        srt_lines.append(f"{convert_time(start)} --> {convert_time(end)}")
        srt_lines.append(f"{sentence}")
        srt_lines.append("")

    return "\n".join(srt_lines)


def transcription_to_srt(transcript: list[TranscriptionResult]) -> str:
    """
    Convert a Speechmatics transcript to SRT subtitle format.

    Args:
        transcript: List of TranscriptionResult objects.

    Returns:
        SRT-formatted subtitle string.
    """
    timings, sentences, _ = transcript_to_sequence(transcript)
    return generate_srt(timings, sentences)


def transcription_to_text(transcript: list[TranscriptionResult]) -> str:
    """
    Convert a Speechmatics transcript to plain text.

    Args:
        transcript: List of TranscriptionResult objects.

    Returns:
        Plain text string with sentences joined by newlines.
    """
    _, sentences, _ = transcript_to_sequence(transcript)
    return "\n".join(sentences)


def transcription_to_json(transcript: list[TranscriptionResult]) -> str:
    """
    Convert a Speechmatics transcript to JSON string.

    Args:
        transcript: List of TranscriptionResult objects.

    Returns:
        JSON string representation of the transcript.
    """
    return json.dumps(transcript)
