'''
Feature Extraction
=================
Converts a list of raw log events (a time window) into a 
fixed-size numerical feature vector for Isolation Forest.

Why a separate module?
Feature engineering is the most important - and most iterable - 
part of ML pipelines. Keeping it isolated means you can improve
features without touching the consumer or model logic. 
'''

import numpy as np


# These are the features we extract - ORDER MATTERS.
# Isolation Forest receives a numpy array with columns in this order.
# Document it clearly so future you don't have to reverse engineer it.
FEATURE_NAMES = [
    "event_count",
    "error_rate",
    "warn_rate",
    "avg_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "status_5xx_rate",
    "status_4xx_rate",
    "latency_std",          # high std = erratic response times = suspicious
    "error_burst_ratio",    # max errors in any 5s sub-window / total errors
] 


def extract_features(events: list[dict]) -> dict:
    """
    Extract numerical features from a window of log events.

    Args:
        events: List of log event dicts from the producer 

    Returns:
        Dict mapping feature names to values.
        Also includes derived stats used for storing in AnomalyAlert.

    Why return a dict instead of a numpy array directly?
    Dicts are readable and debuggable. We convert to array only 
    when feeding into sklearn - keeping the two concerns separate
    """
    if not events:
        # Return a zero-vector if the window is empty.
        # This shouldn't happen in practice but prevents crashes.
        return {name: 0.0 for name in FEATURE_NAMES}
    
    n = len(events)     # total events in the window

    # ── Level counts ─────────────────────────────────────────────
    levels      = [e["level"] for e in events]
    error_count = levels.count("ERROR")
    warn_count  = levels.count("WARN")
    info_count  = levels.count("INFO")

    error_rate  = error_count / n
    warn_rate   = warn_count / n

    # ── Latency stats ───────────────────────────────────────────
    latencies = [e["latency_ms"] for e in events]
    avg_latency    = float(np.mean(latencies))
    std_latency    = float(np.std(latencies))
    p95_latency    = float(np.percentile(latencies, 95))
    p99_latency    = float(np.percentile(latencies, 99))

    # ── HTTP status code breakdown ──────────────────────────────
    status_codes   = [e["status_code"] for e in events]
    status_5xx     = sum(1 for s in status_codes if 500 <= s < 600)
    status_4xx     = sum(1 for s in status_codes if 400 <= s < 500)
    status_5xx_rate = status_5xx / n
    status_4xx_rate = status_4xx / n

    # ── Error burst ratio ─────────────────────────────────────────
    # Detects concentrated error spikes within a window.
    # A sustained 20% error rate is less alarming than 80% for 5s then 0%.
    # We split the window into 6 x 5-second sub-buckets and find the max.
    error_burst_ratio = _compute_burst_ratio(events, error_count)

    return {
        # Core features for Isolation Forest
        "event_count":       float(n),
        "error_rate":        error_rate,
        "warn_rate":         warn_rate,
        "avg_latency_ms":    avg_latency,
        "p95_latency_ms":    p95_latency,
        "p99_latency_ms":    p99_latency,
        "status_5xx_rate":   status_5xx_rate,
        "status_4xx_rate":   status_4xx_rate,
        "latency_std":       std_latency,
        "error_burst_ratio": error_burst_ratio,

        # Extra stats stored in AnomalyAlert but NOT fed to the model
        # (they're either redundant or non-numeric-safe)
        "_error_count":      error_count,
        "_warn_count":       warn_count,
        "_info_count":       info_count,
    }

def _compute_burst_ratio(events: list[dict], total_errors: int) -> float:
    """
    Find the maximum error concentration in any 5-second sub-window.

    Returns the fraction of total errors that occurred in the worst 5s.
    A value near 1.0 means all errors happened at once — a burst.
    A value near 0.16 (1/6) means errors were evenly distributed.
    """
    if total_errors == 0:
        return 0.0

    # Parse timestamps and bucket by 5-second intervals
    from datetime import datetime

    buckets: dict[int, int] = {}
    for event in events:
        if event["level"] == "ERROR":
            # Convert ISO timestamp to Unix epoch, then floor to 5s bucket
            ts = datetime.fromisoformat(
                event["timestamp"].replace("Z", "+00:00")
            )
            bucket_key = int(ts.timestamp()) // 5
            buckets[bucket_key] = buckets.get(bucket_key, 0) + 1

    if not buckets:
        return 0.0

    max_in_bucket = max(buckets.values())
    return max_in_bucket / total_errors


def features_to_array(feature_dict: dict) -> np.ndarray:
    """
    Convert a feature dict to a numpy array for sklearn.

    Extracts only the FEATURE_NAMES keys (not the _ prefixed extras).
    Shape: (1, n_features) — sklearn expects 2D arrays even for 1 sample.
    """
    vector = [feature_dict[name] for name in FEATURE_NAMES]
    return np.array(vector).reshape(1, -1)