# Chat

Chat stores persistent user conversations with AI.

## Responsibilities

- `ChatSession`: user conversation container.
- `ChatThread`: a branch/model context inside a session.
- `ChatMessage`: user, assistant, or system messages inside a thread.
- Generate assistant replies through the shared provider client.
- Store provider metadata and finance usage in assistant message metadata.

The OpenAI-compatible proxy is intentionally not here. It lives in
`apps/language/completion`.
