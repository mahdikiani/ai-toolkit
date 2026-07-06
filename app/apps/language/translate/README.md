# Translate

Translate is a specialized text tool backed by Promptic.

## Responsibilities

- Accept source text and target language.
- Render the `translate` prompt.
- Call the configured LLM provider.
- Store translated output, provider metadata, and finance usage.

## API

- `GET /api/ai/v1/translates`
- `POST /api/ai/v1/translates`
- `GET /api/ai/v1/translates/{uid}`
- `GET /api/ai/v1/translates/{uid}/result`
