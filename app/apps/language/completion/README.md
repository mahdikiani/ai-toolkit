# Completion

Completion exposes OpenAI-compatible proxy endpoints.

## Responsibilities

- Authenticate requests with USSO.
- Proxy `/v1/chat/completions` style requests to configured providers.
- Support streaming and non-streaming responses.
- Preserve response shape for clients.
- Meter provider usage when non-streaming provider metadata is available.

## API

- `POST /api/ai/v1/chat/completions`
