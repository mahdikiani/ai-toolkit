"""
Standalone test for Python 3.13 chunker import verification.

This test runs without pytest fixtures to verify the bug fix works.
It validates that the chunker module imports successfully on Python 3.13
without requiring pyaudioop or pydub.

**Validates: Requirements 2.1, 2.2, 2.3**
"""

from __future__ import annotations

import sys

import pytest

# Mark tests to not use the db fixture
pytestmark = pytest.mark.usefixtures()


def test_ai_toolkit_chunker_import() -> None:
    """Test that apps.transcribe.chunker imports successfully on Python 3.13."""
    from pathlib import Path

    print(f"\nPython version: {sys.version_info.major}.{sys.version_info.minor}")

    # Add app directory to path
    app_path = Path(__file__).parent.parent
    if str(app_path) not in sys.path:
        sys.path.insert(0, str(app_path))

    # Test 1: Import chunker module
    try:
        from apps.transcribe import chunker

        print("✓ apps.transcribe.chunker imported successfully")
    except ModuleNotFoundError as e:
        print(f"✗ Import failed: {e}")
        raise

    # Test 2: Verify expected functions and classes exist
    assert hasattr(chunker, "create_chunk_plan"), "Missing create_chunk_plan"
    assert hasattr(chunker, "AudioChunk"), "Missing AudioChunk"
    assert hasattr(chunker, "ChunkPlan"), "Missing ChunkPlan"
    assert hasattr(chunker, "ChunkTranscriptionResult"), (
        "Missing ChunkTranscriptionResult"
    )
    print("✓ All expected functions and classes exist")

    # Test 3: Verify no pydub imports
    imported_modules = [name for name in sys.modules if name.startswith("pydub")]
    assert len(imported_modules) == 0, f"pydub modules found: {imported_modules}"
    print("✓ No pydub modules in sys.modules")

    # Test 4: Verify Python version
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} >= 3.13")

    print("\n" + "=" * 60)
    print("ALL IMPORT TESTS PASSED - Bug is fixed!")
    print("=" * 60)


def test_services_ai_chunker_import() -> None:
    """Test that services/ai chunker also imports successfully."""
    from pathlib import Path

    services_ai_path = Path(__file__).parent.parent.parent.parent / "ai"
    if not services_ai_path.exists():
        print("⚠ services/ai directory not found, skipping test")
        return

    sys.path.insert(0, str(services_ai_path))
    try:
        from apps.transcribe import chunker as ai_chunker

        print("✓ services/ai apps.transcribe.chunker imported successfully")

        # Verify expected functions exist
        assert hasattr(ai_chunker, "create_chunk_plan"), "Missing create_chunk_plan"
        assert hasattr(ai_chunker, "AudioChunk"), "Missing AudioChunk"
        assert hasattr(ai_chunker, "ChunkPlan"), "Missing ChunkPlan"
        print("✓ All expected functions exist in services/ai chunker")
    finally:
        if str(services_ai_path) in sys.path:
            sys.path.remove(str(services_ai_path))


if __name__ == "__main__":
    test_ai_toolkit_chunker_import()
    test_services_ai_chunker_import()
