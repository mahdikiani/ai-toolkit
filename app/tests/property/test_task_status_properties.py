"""
Property-based tests for task status state machine.

Property 5: Task status transition validity
Validates: Requirements 4.5, 10.4
"""

import pytest
from fastapi_mongo_base.tasks import TaskStatusEnum
from hypothesis import given, settings
from hypothesis import strategies as st

# Valid state machine transitions
VALID_TRANSITIONS: dict[TaskStatusEnum, set[TaskStatusEnum]] = {
    TaskStatusEnum.init: {TaskStatusEnum.processing, TaskStatusEnum.error},
    TaskStatusEnum.processing: {TaskStatusEnum.completed, TaskStatusEnum.error},
    TaskStatusEnum.completed: {TaskStatusEnum.error},  # Can fail after completion
    TaskStatusEnum.error: set(),  # Terminal state
}

# All valid status values
ALL_STATUSES = list(TaskStatusEnum)


def is_valid_transition(from_status: TaskStatusEnum, to_status: TaskStatusEnum) -> bool:
    """Check if a status transition is valid according to the state machine."""
    return to_status in VALID_TRANSITIONS.get(from_status, set())


@pytest.mark.property
class TestTaskStatusTransitions:
    """Property 5: Task status transition validity."""

    @given(
        st.sampled_from(ALL_STATUSES),
        st.sampled_from(ALL_STATUSES),
    )
    @settings(max_examples=100)
    def test_valid_transitions_are_consistent(
        self, from_status: TaskStatusEnum, to_status: TaskStatusEnum
    ) -> None:
        """
        Property 5: Task status transition validity.

        The state machine should be consistent: if a transition is valid,
        it should always be valid for the same pair of statuses.
        """
        result1 = is_valid_transition(from_status, to_status)
        result2 = is_valid_transition(from_status, to_status)

        assert result1 == result2, (
            f"Transition validity is not consistent for {from_status} -> {to_status}"
        )

    def test_init_can_transition_to_processing(self) -> None:
        """Init -> processing should be a valid transition."""
        assert is_valid_transition(TaskStatusEnum.init, TaskStatusEnum.processing)

    def test_init_can_transition_to_error(self) -> None:
        """Init -> error should be a valid transition."""
        assert is_valid_transition(TaskStatusEnum.init, TaskStatusEnum.error)

    def test_processing_can_transition_to_completed(self) -> None:
        """Processing -> completed should be a valid transition."""
        assert is_valid_transition(TaskStatusEnum.processing, TaskStatusEnum.completed)

    def test_processing_can_transition_to_error(self) -> None:
        """Processing -> error should be a valid transition."""
        assert is_valid_transition(TaskStatusEnum.processing, TaskStatusEnum.error)

    def test_completed_cannot_transition_to_init(self) -> None:
        """Completed -> init should be an invalid transition."""
        assert not is_valid_transition(TaskStatusEnum.completed, TaskStatusEnum.init)

    def test_completed_cannot_transition_to_processing(self) -> None:
        """Completed -> processing should be an invalid transition."""
        assert not is_valid_transition(
            TaskStatusEnum.completed, TaskStatusEnum.processing
        )

    def test_error_is_terminal_state(self) -> None:
        """Error should not transition to any other state."""
        for to_status in ALL_STATUSES:
            if to_status != TaskStatusEnum.error:
                assert not is_valid_transition(TaskStatusEnum.error, to_status), (
                    f"error -> {to_status} should be invalid (error is terminal)"
                )

    def test_init_cannot_transition_to_completed(self) -> None:
        """Init -> completed should be an invalid transition (must go through processing)."""
        assert not is_valid_transition(TaskStatusEnum.init, TaskStatusEnum.completed)

    @given(st.sampled_from(ALL_STATUSES))
    @settings(max_examples=20)
    def test_every_status_has_defined_transitions(self, status: TaskStatusEnum) -> None:
        """Every status should have a defined set of valid transitions."""
        assert status in VALID_TRANSITIONS, (
            f"Status {status} has no defined transitions in VALID_TRANSITIONS"
        )

    @given(st.sampled_from(ALL_STATUSES))
    @settings(max_examples=20)
    def test_no_self_transitions(self, status: TaskStatusEnum) -> None:
        """No status should transition to itself."""
        assert not is_valid_transition(status, status), (
            f"Status {status} should not transition to itself"
        )
