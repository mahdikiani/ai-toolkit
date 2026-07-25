# AI Toolkit

FastAPI backend (Python 3.12/3.13, managed by `uv`) exposing AI modules (OCR, Transcribe, YouTube, Chat, Promptic, Translate, Webpage, OpenAI-compat) under the base path `/api/ai/v1`. Persistence is MongoDB via Beanie. Auth is external USSO (JWT verified against `usso.uln.me` JWKS). See `README.md` and `app/README.md` for module/command details.

## Cursor Cloud specific instructions

Single service: the FastAPI app in `app/`. The startup update script already runs `uv sync --dev --directory app`, so dependencies are installed on boot.

### Environment prerequisites (baked into the VM snapshot)
- `uv` lives at `~/.local/bin` (already on `PATH` via `~/.bashrc`). Run commands from `app/` or use `uv --directory app ...`.
- System packages installed: `poppler-utils`, `ffmpeg`, `libmagic1` (needed by OCR/transcribe/mime code), plus a local `mongodb-org` (v8) server.
- A dev `.env` exists at the repo root (`/workspace/.env`, copied from `sample.env`) with `MONGO_URI=mongodb://localhost:27017/`. `.env` is git-ignored. `dotenv` resolves this root `.env` for both the app and tests because it walks up from `app/`.

### Running the app
- MongoDB is NOT auto-started. Start it first, then the app:
  - `mongod --dbpath /var/lib/mongodb --bind_ip 127.0.0.1 --port 27017` (run in a tmux/background session).
  - `cd app && uv run python main.py` — serves on `http://localhost:8000`; Swagger UI at `/api/ai/v1/docs`, liveness `/api/ai/v1/health`, readiness `/api/ai/v1/ready`.
- The app fails fast on startup if MongoDB is unreachable (it is not optional). If you see `MongoDBConnectionError` on boot, mongod is not running.

### Auth caveat
- Every write/task endpoint requires a valid USSO JWT verified against the external `usso.uln.me` JWKS; there is no local token minting. To exercise write paths + persistence without a token, drive the app's own model/service layer directly (e.g. `WebpageTask.create_item(...)` then `.start_processing()`), which the Webpage module supports without external LLM keys (it fetches via Jina Reader, `https://r.jina.ai/`). LLM-backed modules (Chat, Promptic, Translate, OpenAI-compat, OCR VLM) additionally need `OPENROUTER_API_KEY`/provider keys in `.env`.

### Lint / test
- Lint: `cd app && uv run ruff check .`. NOTE: ruff config has `fix = true`, so `ruff check` auto-modifies files. Use `uv run ruff check . --no-fix` (or revert with `git checkout -- app/`) if you only want to inspect.
- Tests: `cd app && uv run pytest`. CI intends marker runs (`-m unit`, `-m integration`, `-m property`) — see `.github/workflows/tests.yml`.
- Known pre-existing failures (unrelated to environment setup): the `tests.yml` workflow only triggers on `ai-toolkit/app/**` paths but this repo's code is under `app/**`, so the test suite never runs in CI and has drifted. `-m property` passes fully; `test_transcribe_services.py` unit tests fail (mock uses `save_report` while code calls `update_and_emit`); integration + `test_db`/`test_health` error because `mongomock-motor` 0.0.36 rejects the `comment` kwarg that beanie 2.1.0 passes to `list_collection_names`.
