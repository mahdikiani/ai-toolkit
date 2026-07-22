# AI Toolkit

Backend سرویس هوش مصنوعی — FastAPI + MongoDB/Beanie. ماژول‌ها به صورت REST API در دسترس هستند، احراز هویت از طریق USSO انجام می‌شود.

## API Base

```
/api/ai/v1
```

## ماژول‌ها

### 🖼️ OCR / Document Intelligence
سه موتور OCR:

| Engine | روش | توضیح |
|--------|-----|-------|
| `pipeline` (پیش‌فرض) | Layout Detection + VLM | تشخیص layout با PP-DocLayoutV2+V3 ensemble، dedup با IOU >40%، استخراج عنصر به عنصر |
| `document_intelligence` (جدید) | AST-based | Pipeline کامل: Loader → Layout → Element Processing (6 VLM مختلف) → AST → Markdown + DOCX |
| `paddleocr_vl_1_5` | PaddleOCR محلی | پردازش تصویر کامل روی CPU |
| `llm` | VLM تمام‌صفحه | ارسال کل صفحه به مدل بینایی |

**Document Intelligence Pipeline (در حال توسعه):**

```
Document Loader → Layout Detection (V2+V3)
→ Element Processing (text/table/formula/figure/chart/code)
→ Reading Order (RTL-aware columns)
→ Document AST → Markdown + DOCX (با OMML برای فرمول‌ها)
```

ویژگی‌ها:
- **تشخیص dual-model** با PP-DocLayoutV2 و V3 هم‌زمان، ادغام با IOU dedup
- **Font Detection** خودکار از PDF → B Nazanin (فارسی) / Calibri (انگلیسی)
- **خروجی DOCX واقعی** با OMML برای فرمول‌ها، Table Grid برای جداول، Header/Footer
- **Asset Manager** برای تصاویر

### 🎤 Transcribe
تبدیل صوت به متن با Soniox API. پشتیبانی از chunking خودکار برای فایل‌های طولانی.

### 📺 YouTube
استخراج زیرنویس و transcript.

### 💬 Chat
Session/Thread/Message با history و SSE streaming.

### 🤖 Promptic
اجرای قالب‌های YAML+Jinja2 با مدل‌های LLM مختلف.

### 🔄 Translate
ترجمه متون از طریق Promptic.

### 🔗 OpenAI Compat
پروکسی `/openai/v1/chat/completions` با billing یکپارچه.

## شروع سریع

```bash
cp sample.env .env   # تنظیم کلیدها
docker compose up -d --build
```

## Development

```bash
cd app
uv sync
uv run ruff check .
uv run mypy .
uv run pytest
```

## Environment Variables

| متغیر | پیش‌فرض | توضیح |
|-------|---------|-------|
| `OCR_ENGINE` | `pipeline` | موتور پیش‌فرض OCR |
| `OCR_VLM_MODEL` | `google/gemini-3.1-flash-lite` | مدل VLM برای OCR |
| `OCR_PIPELINE_DPI` | `300` | رزولوشن صفحات PDF |
| `DEFAULT_MODEL` | `openai/gpt-4o-mini` | مدل پیش‌فرض Chat |
| `OPENROUTER_API_KEY` | — | کلید API برای OpenRouter |
