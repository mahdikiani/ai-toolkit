# AI Toolkit

Backend سرویس هوش مصنوعی — FastAPI + MongoDB/Beanie. ماژول‌ها به صورت REST API در دسترس هستند، احراز هویت از طریق USSO انجام می‌شود.

## API Base

```
/api/ai/v1
```

## ماژول‌ها

### 🖼️ OCR / Document Intelligence
موتورهای OCR (انتخاب با `OCR_ENGINE` یا فیلد `ocr_engine` در درخواست؛ درخواست اولویت دارد):

| Engine | روش | توضیح |
|--------|-----|-------|
| `pipeline` (پیش‌فرض) | Layout Detection + VLM | تشخیص layout با PP-DocLayoutV2+V3 ensemble، dedup با IOU >40%، استخراج عنصر به عنصر |
| `document_intelligence` | AST-based | Pipeline کامل: Loader → Layout → Element Processing → AST → Markdown + DOCX |
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
وبهوک‌های Soniox با `SONIOX_WEBHOOK_SECRET` امضا می‌شوند.

### 📺 YouTube
استخراج زیرنویس و transcript.

### 💬 Chat
Session/Thread/Message با history و SSE streaming.

### 🤖 Promptic
اجرای قالب‌های YAML+Jinja2 با مدل‌های LLM مختلف.

### 🔄 Translate
ترجمه متون از طریق Promptic.

### 🔗 OpenAI Compat (سطح زنده)
پروکسی سازگار با OpenAI در `/openai/v1`:

| Endpoint | توضیح |
|----------|--------|
| `GET /openai/v1/models` | لیست مدل‌ها |
| `POST /openai/v1/chat/completions` | چت (stream و non-stream با metering) |
| `POST /openai/v1/audio/speech` | TTS از طریق OpenRouter |
| `POST /openai/v1/audio/transcriptions` | رونویسی sync با Soniox |

ماژول `language/completion` همان منطق مشترک را reuse می‌کند و مسیر canonical همین `/openai/v1` است.

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
| `OCR_ENGINE` | `pipeline` | موتور پیش‌فرض OCR (`pipeline` یا `document_intelligence` و…) |
| `OCR_VLM_MODEL` | `google/gemini-3.1-flash-lite` | مدل VLM برای OCR |
| `OCR_PIPELINE_DPI` | `300` | رزولوشن صفحات PDF |
| `DEFAULT_MODEL` | `openai/gpt-4o-mini` | مدل پیش‌فرض Chat |
| `DEFAULT_TTS_MODEL` | `openai/gpt-4o-mini-tts` | مدل TTS برای `/openai/v1/audio/speech` |
| `OPENAI_COMPAT_MODELS` | — | لیست مدل‌های اضافه برای `/openai/v1/models` (با کاما) |
| `OPENROUTER_API_KEY` | — | کلید API برای OpenRouter |
| `SONIOX_API_KEY` | — | کلید Soniox برای transcribe |
| `SONIOX_WEBHOOK_SECRET` | — | امضای HMAC وب‌هوک Soniox (الزامی برای callback) |
