# Transcribe

Transcribe converts audio/video input to text.

## Responsibilities

- Accept audio or video file URLs.
- Accept direct multipart uploads without using an external media service.
- Accept base64 uploads as data URLs.
- Select transcription provider/model, defaulting to Soniox.
- Chunk long audio with ffmpeg when enabled.
- Process provider webhooks.
- Store transcript, provider metadata, and per-minute finance usage.

## Notes

Video handling belongs inside this app: video inputs should be normalized to audio
before sending to a transcription provider. Generic download helpers live under
`utils/downloaders` for future sources like Google Drive.

## API

- `POST /api/ai/v1/transcribes`
- `POST /api/ai/v1/transcribes/upload/file`
- `POST /api/ai/v1/transcribes/upload/base64`
- `GET /api/ai/v1/transcribes/{uid}/result`
