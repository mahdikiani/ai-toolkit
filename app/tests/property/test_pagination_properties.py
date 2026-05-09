"""
Property-based tests for pagination logic.

Property 4: Pagination invariants
Validates: Requirements 4.4, 2.8
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st


def paginate(items: list, offset: int, limit: int) -> tuple[list, int]:
    """Simulate pagination logic."""
    total = len(items)
    page = items[offset : offset + limit]
    return page, total


@pytest.mark.property
class TestPaginationInvariants:
    """Property 4: Pagination invariants."""

    @given(
        st.lists(st.integers(), max_size=100),
        st.integers(min_value=0, max_value=200),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_returned_count_never_exceeds_limit(
        self, items: list, offset: int, limit: int
    ) -> None:
        """Property 4a: Returned item count should never exceed the limit."""
        page, _total = paginate(items, offset, limit)

        assert len(page) <= limit, f"Returned {len(page)} items but limit is {limit}"

    @given(
        st.lists(st.integers(), min_size=1, max_size=100),
        st.integers(min_value=0, max_value=99),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_offset_plus_count_never_exceeds_total(
        self, items: list, offset: int, limit: int
    ) -> None:
        """Property 4b: offset + returned_count should never exceed total."""
        assume(offset < len(items))

        page, total = paginate(items, offset, limit)

        assert offset + len(page) <= total, (
            f"offset ({offset}) + returned_count ({len(page)}) > total ({total})"
        )

    @given(
        st.lists(st.integers(), max_size=100),
        st.integers(min_value=0, max_value=200),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_empty_result_when_offset_exceeds_total(
        self, items: list, offset: int, limit: int
    ) -> None:
        """Property 4c: Should return empty list when offset >= total."""
        total = len(items)
        assume(offset >= total)

        page, _ = paginate(items, offset, limit)

        assert len(page) == 0, (
            f"Expected empty page when offset ({offset}) >= total ({total}), "
            f"but got {len(page)} items"
        )

    @given(
        st.lists(st.integers(), min_size=1, max_size=100),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_all_items_covered_by_full_pagination(
        self, items: list, limit: int
    ) -> None:
        """Property 4d: Paginating through all pages should cover all items exactly once."""
        total = len(items)
        all_collected = []
        offset = 0

        while offset < total:
            page, _ = paginate(items, offset, limit)
            if not page:
                break
            all_collected.extend(page)
            offset += len(page)

        assert all_collected == items, (
            "Full pagination did not cover all items exactly once"
        )

    @given(
        st.lists(st.integers(), max_size=100),
        st.integers(min_value=0, max_value=200),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_total_is_always_non_negative(
        self, items: list, offset: int, limit: int
    ) -> None:
        """Property 4e: Total count should always be non-negative."""
        _, total = paginate(items, offset, limit)

        assert total >= 0, f"Total count is negative: {total}"

    @given(
        st.lists(st.integers(), max_size=100),
        st.integers(min_value=0, max_value=200),
        st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_total_is_independent_of_offset_and_limit(
        self, items: list, offset: int, limit: int
    ) -> None:
        """Property 4f: Total count should not depend on offset or limit."""
        _, total1 = paginate(items, offset, limit)
        _, total2 = paginate(items, 0, 1)  # Different offset and limit

        assert total1 == total2, (
            f"Total count changed with different offset/limit: {total1} vs {total2}"
        )
