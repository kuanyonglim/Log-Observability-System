"""
Tests for feature extraction logic.

Strategy: test each feature independently so failures pinpoint
exactly which calculation broke — not just "features are wrong".
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'consumer'))

from features import (
    extract_features,
    features_to_array,
    FEATURE_NAMES,
)


class TestExtractFeatures:
    """Tests for the extract_features() function."""

    def test_returns_all_expected_keys(self, normal_log_events):
        """Every FEATURE_NAMES key must be present in output."""
        result = extract_features(normal_log_events)
        for name in FEATURE_NAMES:
            assert name in result, f"Missing feature key: {name}"

    def test_error_rate_calculation(self, fault_log_events):
        """
        error_rate = ERROR events / total events.
        Fault window has 30 ERROR + 5 INFO = 35 total.
        Expected error_rate = 30/35 ≈ 0.857
        """
        result = extract_features(fault_log_events)
        expected = 30 / 35
        assert abs(result["error_rate"] - expected) < 0.01, (
            f"Expected error_rate ≈ {expected:.3f}, got {result['error_rate']:.3f}"
        )

    def test_normal_window_low_error_rate(self, normal_log_events):
        """Normal window should have < 10% error rate."""
        result = extract_features(normal_log_events)
        assert result["error_rate"] < 0.10, (
            f"Normal window error_rate too high: {result['error_rate']}"
        )

    def test_fault_window_high_latency(self, fault_log_events):
        """Fault window p99 latency should exceed 3000ms."""
        result = extract_features(fault_log_events)
        assert result["p99_latency_ms"] > 3000, (
            f"Expected p99 > 3000ms, got {result['p99_latency_ms']}"
        )

    def test_p99_greater_than_p95(self, normal_log_events):
        """
        p99 must always be >= p95 by definition.
        If this fails, our percentile calculation is broken.
        """
        result = extract_features(normal_log_events)
        assert result["p99_latency_ms"] >= result["p95_latency_ms"], (
            "p99 must be >= p95"
        )

    def test_status_5xx_rate_correct(self, fault_log_events):
        """
        Fault events have 30 x 503 (5xx) out of 35 total.
        5xx rate should be ≈ 0.857
        """
        result = extract_features(fault_log_events)
        expected = 30 / 35
        assert abs(result["status_5xx_rate"] - expected) < 0.01

    def test_empty_window_returns_zeros(self):
        """
        Empty window should not crash — return zero vector.
        This can happen if a service goes silent for 30 seconds.
        """
        result = extract_features([])
        for name in FEATURE_NAMES:
            assert result[name] == 0.0, f"Expected 0.0 for {name}, got {result[name]}"

    def test_event_count_matches_input_length(self, normal_log_events):
        """event_count must exactly equal len(events)."""
        result = extract_features(normal_log_events)
        assert result["event_count"] == float(len(normal_log_events))

    def test_error_burst_ratio_bounded(self, fault_log_events):
        """error_burst_ratio must be between 0.0 and 1.0 inclusive."""
        result = extract_features(fault_log_events)
        assert 0.0 <= result["error_burst_ratio"] <= 1.0, (
            f"burst_ratio out of bounds: {result['error_burst_ratio']}"
        )


class TestFeaturesToArray:
    """Tests for the features_to_array() conversion function."""

    def test_output_shape_is_2d(self, normal_features):
        """sklearn expects shape (1, n_features) — a 2D array."""
        arr = features_to_array(normal_features)
        assert arr.ndim == 2, f"Expected 2D array, got {arr.ndim}D"
        assert arr.shape[0] == 1, f"Expected 1 row, got {arr.shape[0]}"

    def test_output_has_correct_number_of_features(self, normal_features):
        """Array width must match FEATURE_NAMES length."""
        arr = features_to_array(normal_features)
        assert arr.shape[1] == len(FEATURE_NAMES), (
            f"Expected {len(FEATURE_NAMES)} features, got {arr.shape[1]}"
        )

    def test_private_keys_excluded(self, normal_features):
        """
        Keys prefixed with _ (like _error_count) are for storage only.
        They must NOT be included in the ML feature vector.
        """
        arr = features_to_array(normal_features)
        # If private keys were included, shape would be wider
        assert arr.shape[1] == len(FEATURE_NAMES)

    def test_values_are_floats(self, normal_features):
        """Isolation Forest requires float dtype — not int."""
        import numpy as np
        arr = features_to_array(normal_features)
        assert arr.dtype in [np.float32, np.float64], (
            f"Expected float array, got {arr.dtype}"
        )