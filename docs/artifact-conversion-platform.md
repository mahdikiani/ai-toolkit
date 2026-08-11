# Artifact Conversion Platform

Status: Phase 1–3 landed; **from-media conversion tasks** added in `0.1.23`
(`feat/artifact-converter`).

## Layers

1. **`apps/artifacts`** — durable SoR (`media:{uid}`)
2. **`POST /convert`** — sync Artifact→Artifact (client already has `artifact_id`)
3. **`POST /conversions/from-media`** — async task: Media URI → Artifacts → webhook

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

Also accepts Media HTTPS URLs that contain `/f/{uid}` on a Media host.

Task completes with `source_artifact_id`, `result_artifact_id`, `result_storage_uri`,
and emits webhook via TaskMixin when `webhook_url` is set.

## Other APIs

- `POST /api/ai/v1/artifacts`
- `GET /api/ai/v1/artifacts/{uid}`
- `POST /api/ai/v1/convert` — `{artifact_id, target_format}`
- `GET /api/ai/v1/convert/formats`
- `/document-convert/*` — legacy streaming compat for mirza

## OCR

Emits `provider_meta.artifact_id` and temporarily dual-writes `docx_url` until
clients migrate to `convert(artifact_id)` / conversion tasks.
