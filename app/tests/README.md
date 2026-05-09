# AI-Toolkit Test Suite

Comprehensive test suite for the ai-toolkit FastAPI service.

## Structure

```
tests/
├── conftest.py              # Root fixtures (app, client, auth, mock_user)
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_executions_services.py
│   ├── test_chat_services.py
│   ├── test_ocr_services.py
│   ├── test_ocr_archive.py
│   ├── test_transcribe_services.py
│   ├── test_translate_services.py
│   ├── test_prompt_engine.py
│   ├── test_texttools.py
│   ├── test_finance.py
│   ├── test_conditions.py
│   ├── test_mime.py
│   ├── test_file_processors.py
│   └── test_models_schemas.py
├── integration/             # Integration tests (API endpoints)
│   ├── test_executions_routes.py
│   ├── test_chat_routes.py
│   ├── test_ocr_routes.py
│   ├── test_transcribe_routes.py
│   ├── test_translate_routes.py
│   └── test_prompts_routes.py
├── property/                # Property-based tests (Hypothesis)
│   ├── conftest.py          # Hypothesis profiles
│   ├── test_texttools_properties.py
│   ├── test_prompt_engine_properties.py
│   ├── test_chunker_properties.py
│   ├── test_pagination_properties.py
│   └── test_task_status_properties.py
└── fixtures/                # Reusable test fixtures
    ├── mock_fixtures.py     # External service mocks
    ├── task_fixtures.py     # Task creation fixtures
    ├── chat_fixtures.py     # Chat session/thread/message fixtures
    └── file_fixtures.py     # File upload fixtures
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=apps --cov=server --cov=utils --cov-report=html --cov-report=term-missing

# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only property-based tests
pytest -m property

# Run fast tests (exclude slow)
pytest -m "not slow"

# Run specific test file
pytest tests/unit/test_executions_services.py

# Run specific test function
pytest tests/unit/test_executions_services.py::TestCheckSchemas::test_missing_prompt_raises_404

# Run with verbose output
pytest -v

# Run with hypothesis statistics
pytest -m property --hypothesis-show-statistics
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `unit` | Fast, isolated unit tests |
| `integration` | API endpoint tests |
| `property` | Property-based tests (Hypothesis) |
| `slow` | Tests that take >1 second |

## Fixtures

### Root Fixtures (conftest.py)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `client` | session | Async HTTP client for the FastAPI app |
| `authenticated_client` | session | Authenticated HTTP client |
| `mock_user` | function | Mock user data dict |
| `db` | session | Initialized test database (mongomock) |

### Mock Fixtures (fixtures/mock_fixtures.py)

| Fixture | Description |
|---------|-------------|
| `mock_openrouter` | Mocks both streaming and non-streaming OpenRouter calls |
| `mock_openrouter_complete` | Mocks only non-streaming OpenRouter calls |
| `mock_openrouter_stream` | Mocks only streaming OpenRouter calls |
| `mock_finance` | Mocks quota checking and metering |
| `mock_finance_insufficient` | Mocks insufficient quota scenario |
| `mock_media` | Mocks file upload/download |
| `mock_file_system` | Provides tmp_path for file system tests |

### Task Fixtures (fixtures/task_fixtures.py)

| Fixture | Description |
|---------|-------------|
| `ocr_task` | Creates a test OCR task |
| `transcribe_task` | Creates a test transcription task |
| `translate_task` | Creates a test translation task |
| `execution_task` | Creates a test execution task |

### Chat Fixtures (fixtures/chat_fixtures.py)

| Fixture | Description |
|---------|-------------|
| `chat_session` | Creates a test chat session |
| `chat_thread` | Creates a test chat thread |
| `chat_messages` | Creates test messages in a thread |

### File Fixtures (fixtures/file_fixtures.py)

| Fixture | Description |
|---------|-------------|
| `mock_png_bytes` | Minimal valid PNG bytes |
| `mock_image_file` | PNG image as BytesIO |
| `mock_pdf_bytes` | Minimal valid PDF bytes |
| `mock_pdf_content` | PDF as BytesIO |
| `mock_audio_bytes` | Minimal valid WAV bytes |
| `mock_audio_file` | WAV audio as BytesIO |
| `base64_png` | Base64-encoded PNG data URL |
| `base64_pdf` | Base64-encoded PDF data URL |
| `base64_audio` | Base64-encoded WAV data URL |

## Correctness Properties

The test suite validates 6 correctness properties using Hypothesis:

1. **Text normalization idempotence**: `normalize(normalize(x)) == normalize(x)`
2. **Prompt engine rendering determinism**: Same input always produces same output
3. **Audio chunk duration preservation**: Sum of chunk durations equals total duration
4. **Pagination invariants**: `returned_count <= limit`, `offset + count <= total`
5. **Task status transition validity**: Only valid state machine transitions allowed
6. **YAML to JSON schema field preservation**: All YAML fields preserved in JSON schema

## Coverage

Target: 80%+ line coverage across all modules.

```bash
# Generate HTML coverage report
pytest --cov=apps --cov=server --cov=utils --cov-report=html
open htmlcov/index.html
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPOTHESIS_PROFILE` | `dev` | Hypothesis profile (`dev`, `ci`, `fast`) |
| `API_KEY` | `""` | API key for authenticated test client |
| `DEBUGPY` | `False` | Enable remote debugging |
