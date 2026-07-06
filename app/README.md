# AI Toolkit App

This directory contains the FastAPI application.

## Structure

- `server/`: app factory settings and router mounting.
- `apps/ocr`: OCR task API.
- `apps/transcribe`: transcription API.
- `apps/youtube`: YouTube subtitle API.
- `apps/language/promptic`: prompt execution engine and task API.
- `apps/language/translate`: translation API built on promptic.
- `apps/language/chat`: persisted chat sessions, threads, and messages.
- `apps/language/completion`: OpenAI-compatible completion proxy.
- `utils/`: reusable provider, finance, media, downloader, MIME, and text helpers.

## Development

```bash
uv run ruff check .
uv run pytest --cov=. --cov-report=term-missing --cov-fail-under=80 -q
```

## Runtime

```bash
uv run python main.py
```

The app serves API docs at `/api/ai/v1/docs`.
