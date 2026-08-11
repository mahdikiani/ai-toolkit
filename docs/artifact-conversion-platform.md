# Artifact Conversion Platform

Status: Phase 1–4 landed on `feat/artifact-converter` / toolkit `0.1.24`.

## Layers

1. **`apps/artifacts`** — durable SoR (`media:{uid}`)
2. **`POST /convert`** — sync Artifact→Artifact (client already has `artifact_id`)
3. **`POST /conversions/from-media`** — async task: Media URI → Artifacts → webhook
4. **Clients (mirza)** — Convert menu uses `convert(artifact_id)` (no attachment UTF-8 decode)

## Conversion task entrypoints

| Endpoint | Status |
|----------|--------|
| `POST /api/ai/v1/conversions/from-media` | **Implemented** |
| `POST /api/ai/v1/conversions/from-upload` | Deferred (security review) |
| `POST /api/ai/v1/conversions/from-base64` | Deferred (security review) |

### from-media example

```json
{
  "source_uri": "media:FILE_UID",
  "source_format": "markdown",
  "target_format": "pdf",
  "title": "گزارش",
  "webhook_url": "https://example.com/hooks/convert"
}
```

Also accepts Media HTTPS URLs that contain `/f/<uid>` on an allowlisted Media host.

## Other APIs

- `POST /api/ai/v1/artifacts`
- `GET /api/ai/v1/artifacts/{uid}`
- `POST /api/ai/v1/convert` — `{artifact_id, target_format}`
- `GET /api/ai/v1/convert/formats`
- `/document-convert/*` — legacy streaming compat (same converter strategies)

## OCR

Emits `provider_meta.artifact_id`. Temporary `docx_url` dual-write is produced via
Converter (`markdown→docx`), not OCR-owned DOCX render/upload.
