# OCR

OCR converts documents and images to text/Markdown.

## Responsibilities

- Accept file URLs, direct multipart uploads, or base64 uploads.
- Detect file type and choose direct extraction or OCR.
- Support configurable OCR engines.
- Current engines include LLM OCR and PaddleOCR variants.
- Store normalized text result, provider metadata, and per-page finance usage.

## Direction

The target pipeline is:

```text
input -> page normalization -> layout/region extraction -> OCR engine -> Markdown assembly
```

The Markdown output should preserve text, tables, figures, formulas, headers, and
footers as much as the selected engine supports.

## API

- `POST /api/ai/v1/ocrs`
- `POST /api/ai/v1/ocrs/upload`
- `POST /api/ai/v1/ocrs/upload/base64`
- `GET /api/ai/v1/ocrs/{uid}/result`
