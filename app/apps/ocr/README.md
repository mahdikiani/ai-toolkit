# OCR

OCR converts documents and images to text/Markdown.

## Responsibilities

- Accept file URLs or direct uploads.
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
