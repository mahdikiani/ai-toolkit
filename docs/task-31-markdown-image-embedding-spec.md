# Task #31 — Embed images when parsing Markdown for the Convert-menu (Word + PDF)

Status: spec ready for implementation. Not started.
Repo: `ai-toolkit` (this repo). Branch: create `fix/markdown-image-parsing` from `main` at the
commit current when work starts — **verify `git log -1` and `pyproject.toml` version match
production before branching** (see §6, this repo had an unauthorized-branch incident on
2026-08-10; do not reuse or cherry-pick from `feat/typst-pdf-renderer` or
`feat/markdown-image-parsing`, both untrusted and untouched by review).

## 1. Context

mirza-bot's "Convert" menu (Word / PDF / Markdown buttons under an OCR or AI-generated result)
sends the result's flattened Markdown text to this repo's `/document-convert/markdown-to-docx`
and `/document-convert/markdown-to-pdf` endpoints
(`app/apps/ocr/convert_routes.py`). Both call
`parse_markdown()` (`app/apps/ocr/document_intelligence/markdown_parser.py`) to turn that
raw Markdown text back into a `DocumentAST`, then render it with the existing DOCX/PDF renderers.

`markdown_parser.py` currently has **no handling at all** for `![caption](url)` image syntax —
confirmed by reading the file: the block-consumer chain is
`_consume_fence → _consume_formula_block → _consume_heading → _consume_table → _consume_list →
_consume_paragraph` (fallback). An image line matches none of these, falls into
`_consume_paragraph`, and the literal `![caption](url)` text is dumped into the document as a
plain paragraph. Both Word and PDF Convert-menu outputs are affected identically, since both
routes share this one parser.

This is confirmed to be the real, currently-unfixed root cause of the reported bug ("images
don't show up when converting through the Convert menu").

Two things already exist and should be reused, not reinvented:

- `app/apps/ocr/document_intelligence/renderers/markdown.py::rewrite_asset_links()` — called
  from `apps/ocr/services.py` right after OCR/generation, rewrites every local asset path in the
  flattened Markdown to a public URL *before* it is cached/sent to the user. **This means the
  Markdown text that reaches `parse_markdown()` via the Convert-menu already contains
  `![caption](https://...)` with real, publicly-fetchable URLs — not local file paths.**
  This is why the fix is "download the URL," not "resolve a local path."
- `app/utils/downloaders/web.py::download_bytes(url, *, http_timeout=120.0) -> BytesIO` —
  the existing SSRF-safe downloader (`assert_safe_url`, gdrive-aware, redirect-revalidation).
  Reuse this exact function; do not write a new HTTP client.

## 2. Goal (in scope)

`parse_markdown()` recognizes a standalone `![caption](src)` line, fetches the image when
`src` is a `http://`/`https://` URL, and produces a `figure` AST node carrying the image bytes,
so both the DOCX and PDF renderers embed it exactly the way they already do for OCR-pipeline
figures.

That's it. Nothing else about PDF/DOCX rendering changes.

## 3. Hard constraints (read before writing code)

These are non-negotiable, in light of what already went wrong on this exact task once:

1. **No rendering-engine changes.** WeasyPrint (PDF) and python-docx (DOCX) stay exactly as
   they are. No Typst, no LibreOffice, no new PDF/DOCX library. This task only teaches the
   *parser* to produce image nodes; the *renderers* already know how to draw a figure node
   (`renderers/pdf.py::_render_figure_or_chart`, `renderers/docx.py::_render_media_node` /
   the image-adding helper around line 900) — do not rewrite them beyond the one small,
   additive change in §4.3.
2. **No dependency changes** beyond what's already installed (`httpx`, `python-magic` are
   already dependencies; matplotlib/weasyprint stay as-is). If you think you need a new
   dependency, stop and ask — don't add it.
3. **No unrelated refactors.** Touch only the files listed in §4. Don't "clean up" adjacent
   code, don't reformat files you're not editing, don't bump unrelated versions.
4. **Security — this is the part that actually matters:**
   - The `/markdown-to-docx/upload` and `/markdown-to-pdf/upload` endpoints accept a raw
     Markdown file from any authenticated user. That means `src` in `![x](src)` is
     **attacker-controlled** the moment this feature exists.
   - **Never treat a parsed `src` as a local filesystem path.** Do not do
     `Path(src).read_bytes()` or set `ASTNode.asset_path = src` for anything that isn't a
     `http://`/`https://` URL that has passed `assert_safe_url`/`download_bytes`. A crafted
     `![x](/etc/passwd)` or `![x](../../../.env)` must never reach a filesystem read — that
     would be a local-file-inclusion bug embedding server files into a user's downloaded
     PDF/DOCX.
   - Any `src` that is not `http://`/`https://`, or whose fetch fails for any reason
     (timeout, non-2xx, SSRF check rejects it, oversized response, unreadable image bytes),
     must fall back to a **caption-only figure node** (no crash, no broken `<img>`/inline
     shape) — this fallback path already exists and is already tested in
     `tests/unit/test_pdf_renderer.py` for the "missing asset" case; reuse it, don't
     reinvent it.
   - Use a **short timeout** for these fetches (recommend 15s, not the 120s default) — a
     Convert-menu tap is a synchronous user-facing action; one slow/unreachable image must
     not hang the whole conversion.
   - Cap response size (recommend rejecting/discarding anything over ~15MB after download —
     check `len(buffer.getvalue())`) so one huge "image" can't blow up memory or the output
     file.
5. **Don't touch `pipeline.py`.** The real OCR pipeline builds its AST directly (never through
   `markdown_parser.py`) and is unaffected by this task; leave it alone.

## 4. Subtasks

### 4.1 — `ASTNode` gains an in-memory image field (`ast.py`)

Add one new field to `ASTNode` (`app/apps/ocr/document_intelligence/ast.py`):

```python
asset_bytes: bytes | None = None  # raw image bytes, when not backed by a filesystem asset_path
```

Rationale: keeps the fix entirely in-memory — no temp files to create/clean up, and it makes
the security constraint in §3.4 structural rather than a "please remember" comment. A node
either has `asset_path` (trusted, filesystem, existing OCR-pipeline behavior — unchanged) or
`asset_bytes` (new, comes only from a successful `download_bytes()` call) or neither (caption
only, existing fallback behavior — unchanged).

### 4.2 — `parse_markdown()` becomes async and recognizes image lines (`markdown_parser.py`)

- Add a regex for a standalone image line, matching exactly what `renderers/markdown.py`
  emits (`![{caption}]({asset_rel})` on its own line):
  ```python
  _IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
  ```
- `parse_markdown` becomes `async def parse_markdown(text: str, title: str = "") -> DocumentAST`.
  Give it the same block-check priority as fence/formula blocks (checked before the
  quote/heading/table/list chain, since an image line won't collide with any of those but
  should never be swallowed by `_consume_paragraph` first).
- On a match: `alt_text, src = match.group(1), match.group(2)`.
  - If `src` starts with `http://` or `https://`: `await` a helper (new, small,
    private to this module — e.g. `_fetch_image_node(alt_text, src)`) that calls
    `download_bytes(src, http_timeout=15.0)`, enforces the size cap, and returns
    `ASTNode(type=LayoutType.figure, caption=alt_text, asset_bytes=buffer.getvalue())`
    on success.
  - On any exception from the fetch (`httpx.HTTPError`, the SSRF-checker's rejection,
    timeout, oversize) **or** when `src` is not http(s): return
    `ASTNode(type=LayoutType.figure, caption=alt_text)` (caption-only, no asset) —
    log a `logger.warning(...)` (module already has/needs a `logging.getLogger(__name__)`,
    check `renderers/pdf.py` for the existing pattern) but never raise.
- Since `parse_markdown` is now async and does real I/O, don't do it inline in the main
  `while` loop's synchronous consumer chain — resolve it explicitly before/alongside the
  existing `for consume in (...)` dispatch, however reads cleanest given the current loop
  shape. Keep every other consumer (`_consume_fence`, `_consume_table`, etc.) synchronous and
  untouched.

### 4.3 — Renderers read `asset_bytes` before `asset_path`

Two small, additive edits — do not restructure either function beyond this:

- `renderers/pdf.py::_image_data_uri` (or the small wrapper that calls it from
  `_render_figure_or_chart`): if `node.asset_bytes` is set, base64-encode it directly
  (skip the `Path(...).exists()`/`read_bytes()` branch entirely for this case). Guess the
  mime type with `python-magic` (already a dependency) on the bytes, falling back to
  `"image/png"` if detection fails — don't rely on `mimetypes.guess_type` here since there's
  no filename/extension to inspect.
- `renderers/docx.py`'s figure-adding helper (around line ~900,
  `asset_path = node.asset_path; if not asset_path or not Path(asset_path).exists(): ...;
  img_bytes = Path(asset_path).read_bytes()`): same idea — if `node.asset_bytes` is set, use
  it directly as `img_bytes` and skip the path-existence/read branch.

### 4.4 — Wire through `convert_routes.py`

All four route handlers call `parse_markdown(...)` — add `await` at each of the four call
sites now that it's async. No other change needed in this file.

## 5. Tests to add (alongside the fix, not after)

- `markdown_parser.py`: standalone `![alt](https://...)` line → figure node with
  `asset_bytes` populated (mock `download_bytes`), `caption == alt`.
- Non-http(s) `src` (e.g. a local-looking path or `/etc/passwd`) → figure node with
  `asset_bytes is None` and `asset_path` **not** set to the attacker-controlled value —
  this is the test that directly guards the LFI concern in §3.4; make it explicit, e.g.
  `test_non_http_image_src_never_becomes_filesystem_path`.
- Fetch failure (mock `download_bytes` to raise `httpx.HTTPError`/timeout) → caption-only
  figure node, no exception propagates out of `parse_markdown`.
- Oversized response (mock a >15MB buffer) → caption-only fallback, not embedded.
- `renderers/pdf.py` / `renderers/docx.py`: a node with `asset_bytes` set (no `asset_path`)
  renders correctly (extend the existing parametrized tests in
  `tests/unit/test_pdf_renderer.py` rather than duplicating them).
- End-to-end through `convert_routes.py`: POST Markdown containing a real (mocked-download)
  image line to `/markdown-to-docx` and `/markdown-to-pdf`, assert the response actually
  contains embedded image data (not literal `![...]` text).

## 6. Acceptance criteria / Definition of Done

- [ ] All new + existing tests pass (`uv run pytest`), `ruff check` clean.
- [ ] Manual check: send a real OCR'd PDF with at least one image to the bot, use the
      Convert-menu on the result for both Word and PDF — the image must actually appear in
      both outputs.
- [ ] Manual check: `/document-convert/markdown-to-pdf/upload` with a crafted
      `![x](/etc/passwd)` in the uploaded file does **not** embed local file contents
      (returns a caption-only figure, no server error).
- [ ] `git log -1` on `main` before merge matches what's actually deployed
      (`docker exec ai-toolkit-app-1 python -c "..."` version check) — confirm no other
      change landed on `main` since this branch was cut.
- [ ] Version bump per repo convention (`pyproject.toml` + `utils/version.py` + their tests).
- [ ] **Mandatory `fable-advisor` review before merge** — pass it this spec plus the diff.
      Do not merge to `main` or deploy without an explicit Ship verdict reported back to the
      orchestrator, and do not rebuild/redeploy the production container yourself even after
      a Ship — hand it back for the orchestrator to do that step.

## 7. Explicitly out of scope (do not do these even if they seem related)

- Inline images inside a text line (`text ![x](y) more text`) — the renderer side
  (`renderers/markdown.py`) only ever emits images on their own line; matching only that
  shape is sufficient and intentional, not a corner someone cut.
- Any change to how the OCR pipeline (`pipeline.py`) builds its own AST directly.
- Any change to `_render_formula`/LaTeX rendering (already fixed and deployed in 0.1.20).
- Anything involving Typst, LibreOffice, or removing/replacing WeasyPrint.
