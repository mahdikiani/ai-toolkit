# Artifact Conversion Platform

Status: verified on toolkit `0.1.24` (`feat/artifact-converter`); merging to
toolkit `main`. PDF engine this wave: WeasyPrint. Typst remains out of scope.

See also: workspace `AI-BOT-CONVERT-TASK.md` and
`AI-BOT-IMPLEMENTATION-PLAN.md`.

## Layers

1. **`apps/artifacts`** — durable SoR (`media:{uid}`)
2. **`POST /convert`** — sync Artifact→Artifact
3. **`POST /conversions/from-media`** — async Media URI → Artifacts → webhook
4. **mirza Convert** — prefers `convert(artifact_id)` when metadata has it

## Close criteria

1. Merge `feat/artifact-converter` into toolkit `main`
2. Manual/API verify: create markdown Artifact → Convert → Word and PDF
   (confirmed via `convert.manual_verify` logs + downloadable `PK`/`%PDF-` bytes)
3. Mark plan phase 6 / queue C `[x]`

## Non-goals this wave

Typst, upload/base64 conversion entrypoints, wide format matrix.
