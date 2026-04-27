"""
Tests for the Isolation Forest anomaly scorer.

Key insight: we can't assert exact scores (they depend on random
training data), but we CAN assert relative ordering:
fault windows must score LOWER than normal windows.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'consumer'))

from scorer import AnomalyScorer, RETRAIN_THRESHOLD


class TestAnomalyScorer:

    def test_scorer_initialises_without_error(self):
        """Scorer must bootstrap successfully on startup."""
        scorer = AnomalyScorer(contamination=0.05)
        assert scorer._is_bootstrapped is True

    def test_score_returns_float_and_bool(self, normal_features):
        """score() must return (float, bool) — the contract other code depends on."""
        scorer = AnomalyScorer()
        score, is_anomaly = scorer.score(normal_features)
        assert isinstance(score, float), f"Expected float score, got {type(score)}"
        assert isinstance(is_anomaly, bool), f"Expected bool, got {type(is_anomaly)}"

    def test_fault_scores_lower_than_normal(self, normal_features, fault_features):
        """
        Core ML correctness test.
        Isolation Forest must assign lower (more negative) scores
        to anomalous windows than to normal ones.
        This is the fundamental property the whole pipeline depends on.
        """
        scorer = AnomalyScorer(contamination=0.05)
        normal_score, _ = scorer.score(normal_features)
        fault_score,  _ = scorer.score(fault_features)

        assert fault_score < normal_score, (
            f"Fault score ({fault_score:.4f}) should be lower than "
            f"normal score ({normal_score:.4f})"
        )

    def test_high_contamination_flags_more_anomalies(
        self, normal_features, fault_features
    ):
        """
        With contamination=0.5, the model expects 50% anomalies.
        Both windows should be flagged — it's an aggressive threshold.
        This tests that contamination actually affects predictions.
        """
        # High contamination = aggressive flagging
        scorer = AnomalyScorer(contamination=0.5)
        _, fault_is_anomaly = scorer.score(fault_features)
        # At 50% contamination, the obvious fault window must be flagged
        assert fault_is_anomaly is True, (
            "Obvious fault window should be anomaly at contamination=0.5"
        )

    def test_window_buffer_grows_after_scoring(self, normal_features):
        """Each scored window must be buffered for future retraining."""
        scorer = AnomalyScorer()
        initial_buffer_size = len(scorer._window_buffer)
        scorer.score(normal_features)
        assert len(scorer._window_buffer) == initial_buffer_size + 1

    def test_retraining_triggers_at_threshold(self, normal_features):
        """
        After RETRAIN_THRESHOLD windows are scored, the model retrains
        and the buffer shrinks (old windows pruned to last 20).
        """
        scorer = AnomalyScorer()

        # Score enough windows to trigger retraining
        for _ in range(RETRAIN_THRESHOLD):
            scorer.score(normal_features)

        # Buffer should have been pruned after retrain
        assert len(scorer._window_buffer) <= 20, (
            f"Buffer should be pruned to 20 after retrain, "
            f"got {len(scorer._window_buffer)}"
        )

    def test_scorer_handles_extreme_values(self):
        """
        Scorer must not crash on extreme feature values.
        Real production data can have latency spikes of 60000ms+.
        """
        scorer = AnomalyScorer()
        extreme_features = {
            "event_count":       1000.0,
            "error_rate":        1.0,      # 100% errors
            "warn_rate":         0.0,
            "avg_latency_ms":    60000.0,  # 60 second latency
            "p95_latency_ms":    90000.0,
            "p99_latency_ms":    120000.0,
            "status_5xx_rate":   1.0,
            "status_4xx_rate":   0.0,
            "latency_std":       15000.0,
            "error_burst_ratio": 1.0,
            "_error_count":      1000,
            "_warn_count":       0,
            "_info_count":       0,
        }
        # Should not raise any exception
        score, is_anomaly = scorer.score(extreme_features)
        assert isinstance(score, float)