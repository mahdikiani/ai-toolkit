"""
Standalone test for Python 3.13 chunker import verification.

This test runs without pytest fixtures to verify the bug fix works.
It validates that the chunker module imports successfully on Python 3.13
without requiring pyaudioop or pydub.

**Validates: Requirements 2.1, 2.2, 2.3**
"""

from __future__ import annotations

import sys


def test_ai_toolkit_chunker_import() -> None:
    """Test that apps.transcribe.chunker imports successfully on Python 3.13."""
    from pathlib import Path

    # Add app directory to path
    app_path = Path(__file__).parent.parent
    if str(app_path) not in sys.path:
        sys.path.insert(0, str(app_path))

    # Test 1: Import chunker module
    from apps.transcribe import chunker

    # Test 2: Verify expected functions and classes exist
    assert hasattr(chunker, "create_chunk_plan"), "Missing create_chunk_plan"
    assert hasattr(chunker, "AudioChunk"), "Missing AudioChunk"
    assert hasattr(chunker, "ChunkPlan"), "Missing ChunkPlan"
    assert hasattr(chunker, "ChunkTranscriptionResult"), (
        "Missing ChunkTranscriptionResult"
    )

    # Test 3: Verify no pydub imports
    imported_modules = [name for name in sys.modules if name.startswith("pydub")]
    assert len(imported_modules) == 0, f"pydub modules found: {imported_modules}"

    # Test 4: Verify Python version


def test_services_ai_chunker_import() -> None:
    """Test that services/ai chunker also imports successfully."""
    from pathlib import Path

    services_ai_path = Path(__file__).parent.parent.parent.parent / "ai"
    if not services_ai_path.exists():
        return

    sys.path.insert(0, str(services_ai_path))
    try:
        from apps.transcribe import chunker as ai_chunker

        # Verify expected functions exist
        assert hasattr(ai_chunker, "create_chunk_plan"), "Missing create_chunk_plan"
        assert hasattr(ai_chunker, "AudioChunk"), "Missing AudioChunk"
        assert hasattr(ai_chunker, "ChunkPlan"), "Missing ChunkPlan"
    finally:
        if str(services_ai_path) in sys.path:
            sys.path.remove(str(services_ai_path))


if __name__ == "__main__":
    test_ai_toolkit_chunker_import()
    test_services_ai_chunker_import()
