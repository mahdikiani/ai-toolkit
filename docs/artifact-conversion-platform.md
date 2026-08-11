# Artifact Conversion Platform (Phase 1–2)

Status: **Phase 1 + Phase 2 landed** in `ai-toolkit` `0.1.21` (`feat/artifact-converter`).

## What shipped

- **`apps/artifacts`** — durable Artifact SoR (`TenantUserEntity`). Content bytes live in Media; Mongo stores metadata + durable `storage_uri` (`media:{uid}`).
- **`apps/converter`** — Artifact→Artifact conversion graph with a registry. First edges: `markdown→docx`, `markdown→pdf` (WeasyPrint unchanged; DI renderers imported, not redesigned).
- **API**
  - `POST /api/ai/v1/artifacts` — create from JSON `{format, content, title?, source?}`
  - `GET /api/ai/v1/artifacts/{uid}`
  - `POST /api/ai/v1/convert` — `{artifact_id, target_format}` → derived Artifact
  - `GET /api/ai/v1/convert/formats` — registered edges
- **Compat** — `/document-convert/*` still streams DOCX/PDF for mirza; rendering now goes through converter strategy helpers.

## Not in this wave

- Phase 3: OCR producers emitting Artifacts / dropping DOCX ownership
- Phase 4: mirza (and other clients) calling `convert(artifact_id, …)`
- PDF engine switch, DI redesign, wide format matrix
