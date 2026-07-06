# AI Toolkit

AI Toolkit is a FastAPI backend for independent AI tools. Each tool is exposed as
a REST API, uses USSO authentication through `fastapi_mongo_base`, stores user
owned task data in MongoDB/Beanie, and records usage through the finance module.

## Current Tools

- OCR: document/image extraction to Markdown with configurable OCR engines.
- Transcribe: audio/video transcription with provider selection, defaulting to Soniox.
- YouTube: subtitle/transcript retrieval through `youtube-transcript.io`.
- Promptic: reusable prompt-template execution using YAML/JSON/Markdown and Jinja2.
- Translate: specialized translation tool backed by Promptic.
- Chat: persistent AI chat sessions, threads, and messages.
- Completion: OpenAI-compatible completion proxy/aggregator.

## API Base

All routes are mounted under:

```text
/api/ai/v1
```

## Quality Gates

From `app/`:

```bash
uv run ruff check .
uv run pytest --cov=. --cov-report=term-missing --cov-fail-under=80 -q
```

## Configuration

Copy `sample.env` to `.env` and configure provider keys as needed. Optional
pricing can be provided as JSON in `AI_TOOLKIT_PRICING`.
