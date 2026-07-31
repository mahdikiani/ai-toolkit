"""Unit tests for voice morph task processing services."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum

from apps.voicemorph.services import process_voice_morph


def _task(**attrs: object) -> MagicMock:
    task = MagicMock()
    task.uid = "task_1"
    task.user_id = "user_1"
    task.audio_url = "https://example.com/audio.wav"
    task.voice_reference_url = None
    task.pitch_shift = None
    task.speed_factor = None
    for key, value in attrs.items():
        setattr(task, key, value)
    task.save_report = AsyncMock()
    return task


@pytest.mark.unit
class TestProcessVoiceMorph:
    """Tests for process_voice_morph."""

    async def test_insufficient_quota_stops_before_generation(self) -> None:
        """
        Regression check.

        process_voice_morph used to have no pre-flight quota check --
        the Replicate prediction was created and billed with no gate
        preventing a broke user from triggering paid generation.
        """
        task = _task()
        with (
            patch(
                "apps.voicemorph.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("0"),
            ),
            patch(
                "apps.voicemorph.services.create_prediction",
                new_callable=AsyncMock,
            ) as mock_predict,
        ):
            result = await process_voice_morph(task)

        mock_predict.assert_not_awaited()
        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once_with("insufficient_quota")

    async def test_successful_morph_meters_and_completes(self) -> None:
        task = _task()
        with (
            patch(
                "apps.voicemorph.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.voicemorph.services.create_prediction",
                new_callable=AsyncMock,
                return_value={"output": "https://example.com/out.wav"},
            ),
            patch(
                "apps.voicemorph.services.finance.meter_cost",
                new_callable=AsyncMock,
            ) as mock_meter,
        ):
            result = await process_voice_morph(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result_url == "https://example.com/out.wav"
        mock_meter.assert_awaited_once()

    async def test_list_output_uses_first_item(self) -> None:
        task = _task()
        with (
            patch(
                "apps.voicemorph.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.voicemorph.services.create_prediction",
                new_callable=AsyncMock,
                return_value={"output": ["https://example.com/out.wav"]},
            ),
            patch(
                "apps.voicemorph.services.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            result = await process_voice_morph(task)

        assert result.result_url == "https://example.com/out.wav"

    async def test_metering_failure_still_completes_with_result(self) -> None:
        """meter_cost failing must not discard an already-produced result."""
        task = _task()
        with (
            patch(
                "apps.voicemorph.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.voicemorph.services.create_prediction",
                new_callable=AsyncMock,
                return_value={"output": "https://example.com/out.wav"},
            ),
            patch(
                "apps.voicemorph.services.finance.meter_cost",
                new_callable=AsyncMock,
                side_effect=RuntimeError("billing service unreachable"),
            ),
        ):
            result = await process_voice_morph(task)

        assert result.task_status == TaskStatusEnum.completed
        assert result.result_url == "https://example.com/out.wav"

    async def test_prediction_failure_sets_error(self) -> None:
        task = _task()
        with (
            patch(
                "apps.voicemorph.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.voicemorph.services.create_prediction",
                new_callable=AsyncMock,
                side_effect=RuntimeError("replicate down"),
            ),
        ):
            result = await process_voice_morph(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once()
        assert "Voice morph failed" in task.save_report.await_args.args[0]

    async def test_no_output_sets_error(self) -> None:
        task = _task()
        with (
            patch(
                "apps.voicemorph.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.voicemorph.services.create_prediction",
                new_callable=AsyncMock,
                return_value={"output": None},
            ),
        ):
            result = await process_voice_morph(task)

        assert result.task_status == TaskStatusEnum.error
        task.save_report.assert_awaited_once_with("No output from voice morph")

    async def test_forwards_optional_fields_to_replicate(self) -> None:
        task = _task(
            voice_reference_url="https://example.com/ref.wav",
            pitch_shift=2.5,
            speed_factor=1.2,
        )
        with (
            patch(
                "apps.voicemorph.services.finance.check_quota",
                new_callable=AsyncMock,
                return_value=Decimal("999999"),
            ),
            patch(
                "apps.voicemorph.services.create_prediction",
                new_callable=AsyncMock,
                return_value={"output": "https://example.com/out.wav"},
            ) as mock_predict,
            patch(
                "apps.voicemorph.services.finance.meter_cost",
                new_callable=AsyncMock,
            ),
        ):
            await process_voice_morph(task)

        input_data = mock_predict.call_args.args[1]
        assert input_data["voice_url"] == "https://example.com/ref.wav"
        assert input_data["pitch_shift"] == pytest.approx(2.5)
        assert input_data["speed_factor"] == pytest.approx(1.2)
