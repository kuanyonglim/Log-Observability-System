"""
Anomaly Scorer
==============
Wraps Isolation Forest with warm-start bootstrapping.

The challenge: Isolation Forest needs training data before it can score.
But we start with zero data. Solution: bootstrap on synthetic 'normal'
data, then retrain on real data once enough has accumulated.
"""

import logging

import numpy as np
from sklearn.ensemble import IsolationForest

from features import FEATURE_NAMES, features_to_array

logger = logging.getLogger("scorer")

# Retrain the model after this many real windows are collected.
# More data = better baseline understanding of "normal".
RETRAIN_THRESHOLD = 50


class AnomalyScorer:
    """
    Manages the Isolation Forest lifecycle:
    1. Bootstrap on synthetic normal data
    2. Score incoming windows
    3. Periodically retrain on accumulated real data
    """

    def __init__(self, contamination: float = 0.05):
        """
        Args:
            contamination: Expected fraction of anomalies in the data.
                           0.05 = we expect ~5% of windows to be anomalous.
                           This affects the decision threshold internally.
                           Too low → misses anomalies. Too high → false alarms.
        """
        self.contamination   = contamination
        self.model           = IsolationForest(
            n_estimators=100,       # number of isolation trees
                                    # more = more stable scores, slower training
            contamination=contamination,
            random_state=42,        # reproducibility
            n_jobs=-1,              # use all CPU cores for training
        )
        self._window_buffer: list[np.ndarray] = []  # accumulated real windows
        self._is_bootstrapped = False

        # Train on synthetic data immediately so we can score from event #1
        self._bootstrap()

    def _bootstrap(self) -> None:
        """
        Train on synthetic 'normal' data so the model is usable immediately.

        Why synthetic? We can't wait 25 minutes for 50 real windows before
        the scorer works. Instead, we generate plausible normal feature
        vectors based on our knowledge of the system's expected behaviour.
        """
        logger.info("🔧 Bootstrapping Isolation Forest on synthetic data...")

        rng = np.random.default_rng(42)
        n_samples = 200

        # Generate synthetic normal windows.
        # Each row = one synthetic 30s window with realistic normal values.
        # We use domain knowledge here: normal error rates are <5%,
        # latency is 80-300ms, etc.
        synthetic = np.column_stack([
            rng.integers(20, 100, n_samples).astype(float),  # event_count
            rng.uniform(0.0, 0.05, n_samples),               # error_rate
            rng.uniform(0.0, 0.08, n_samples),               # warn_rate
            rng.uniform(80, 300, n_samples),                 # avg_latency_ms
            rng.uniform(200, 600, n_samples),                # p95_latency_ms
            rng.uniform(300, 800, n_samples),                # p99_latency_ms
            rng.uniform(0.0, 0.03, n_samples),               # status_5xx_rate
            rng.uniform(0.0, 0.05, n_samples),               # status_4xx_rate
            rng.uniform(10, 80, n_samples),                  # latency_std
            rng.uniform(0.0, 0.25, n_samples),               # error_burst_ratio
        ])

        self.model.fit(synthetic)
        self._is_bootstrapped = True
        logger.info("✅ Bootstrap complete. Model ready to score.")

    def score(self, feature_dict: dict) -> tuple[float, bool]:
        """
        Score a single feature window.

        Returns:
            (anomaly_score, is_anomaly)
            - anomaly_score: raw IsolationForest score (negative = anomalous)
            - is_anomaly: True if the model predicts this window is anomalous
        """
        X = features_to_array(feature_dict)

        # score_samples returns the anomaly score of each sample.
        # Lower = more anomalous. Typically in range [-0.5, 0.5].
        anomaly_score = float(self.model.score_samples(X)[0])

        # predict returns 1 (normal) or -1 (anomaly)
        prediction    = self.model.predict(X)[0]
        is_anomaly    = prediction == -1

        # Buffer this window for future retraining
        self._window_buffer.append(X.flatten())

        # Retrain once we have enough real data
        if len(self._window_buffer) >= RETRAIN_THRESHOLD:
            self._retrain()

        return anomaly_score, is_anomaly

    def _retrain(self) -> None:
        """
        Retrain Isolation Forest on accumulated real windows.

        Why retrain? The bootstrap data is synthetic. Real traffic patterns
        differ — different time-of-day patterns, different service mix.
        Retraining on real data makes the model adapt to actual baseline.
        """
        logger.info(
            "🔄 Retraining Isolation Forest on %d real windows...",
            len(self._window_buffer)
        )
        X_real = np.array(self._window_buffer)
        self.model.fit(X_real)
        # Clear buffer but keep last 20 windows as warm start for next retrain
        self._window_buffer = self._window_buffer[-20:]
        logger.info("✅ Retrain complete.")