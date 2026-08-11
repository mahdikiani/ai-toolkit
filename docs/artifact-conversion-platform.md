# Artifact Conversion Platform (Phase 1–3)

Status: **Phase 1 through Phase 3 landed** in `ai-toolkit` `0.1.22` (`feat/artifact-converter`).

## What shipped

- **`apps/artifacts`** — durable Artifact SoR (`TenantUserEntity`). Content bytes live in Media; Mongo stores metadata + durable `storage_uri` (`media:{uid}`).
- **`apps/converter`** — Artifact→Artifact conversion graph with a registry. First edges: `markdown→docx`, `markdown→pdf` (WeasyPrint unchanged; DI renderers imported, not redesigned).
- **API**
  - `POST /api/ai/v1/artifacts` — create from JSON `{format, content, title?, source?}`
  - `GET /api/ai/v1/artifacts/{uid}`
  - `POST /api/ai/v1/convert` — `{artifact_id, target_format}` → derived Artifact
  - `GET /api/ai/v1/convert/formats` — registered edges
- **Compat** — `/document-convert/*` still streams DOCX/PDF for mirza; rendering now goes through converter strategy helpers.
- **OCR producers** — pipeline and Document Intelligence OCR persist their final Markdown as an `Artifact` with `source="ocr"` and publish its UID as `provider_meta.artifact_id`. As temporary compatibility for mirza Word delivery, both paths also build/upload DOCX and dual-write `provider_meta.docx_url` until Phase 4 clients use `convert(artifact_id)`.

Patch note (`0.1.22`): completed the Phase 3 OCR ownership migration while temporarily retaining the legacy DOCX URL alongside the Artifact ID for mirza compatibility.

## Not in this wave

- Phase 4: mirza (and other clients) calling `convert(artifact_id, …)`
- PDF engine switch, DI redesign, wide format matrix
