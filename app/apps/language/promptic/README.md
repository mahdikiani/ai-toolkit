# Promptic

Promptic executes reusable prompt templates.

## Responsibilities

- Load prompt files from `prompts/`.
- Render variables with Jinja2.
- Support YAML, JSON, Markdown, text, and TOML prompt inputs.
- Call the configured LLM provider through async HTTP clients.
- Store prompt run status, result, provider metadata, and finance usage.

## API

- `GET /api/ai/v1/promptic`
- `POST /api/ai/v1/promptic?prompt_name=<name>`
- `GET /api/ai/v1/promptic/{uid}`

## Prompt Files

Prompt files can use either structured fields:

```yaml
task:
  system:
    personas: You are helpful.
  user: |
    Process {{ content }}
```

or simpler fields:

```yaml
system: You are helpful.
prompt: Process {{ content }}
```
