# Artifact Conversion Platform

Status: `[x]` — merged to toolkit `main` @ `7456862` / deployed `0.1.24`.
Convert Word/PDF verified (`convert.manual_verify` + downloadable bytes).
PDF engine this wave: WeasyPrint. Typst remains out of scope.

See also: workspace `AI-BOT-CONVERT-TASK.md` and
`AI-BOT-IMPLEMENTATION-PLAN.md`.

## Layers

1. **`apps/artifacts`** — durable SoR (`media:{uid}`)
2. **`POST /convert`** — sync Artifact→Artifact
3. **`POST /conversions/from-media`** — async Media URI → Artifacts → webhook
4. **mirza Convert** — prefers `convert(artifact_id)` when metadata has it

## Closed

1. Merged `feat/artifact-converter` into toolkit `main`
2. Verified: markdown Artifact → Convert → Word and PDF
3. Plan phase 6 / item C marked `[x]`

## Non-goals this wave

Typst, upload/base64 conversion entrypoints, wide format matrix.
