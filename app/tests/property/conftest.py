"""Configuration and fixtures for property-based tests."""

import os

import pytest

hypothesis = pytest.importorskip(
    "hypothesis",
    reason="property tests require the optional hypothesis dependency",
)
HealthCheck = hypothesis.HealthCheck
settings = hypothesis.settings

# Configure Hypothesis profiles
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "fast",
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Load profile based on environment
profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
settings.load_profile(profile)
