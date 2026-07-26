# Completion

Shared helpers for OpenAI-compatible chat completions.

## Canonical surface

Prefer the mounted OpenAI-compat API:

- `POST /api/ai/v1/openai/v1/chat/completions`
- `POST /api/ai/v1/openai/v1/audio/speech`
- `POST /api/ai/v1/openai/v1/audio/transcriptions`

This package reuses `apps.openai_compat.services` for proxy + metering (including streams).
The optional `routes.py` alias (`/chat/completions`) is not mounted by default.
