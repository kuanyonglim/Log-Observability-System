"""
conftest.py
===========
pytest configuration and shared fixtures.

Fixtures defined here are automatically available to ALL test files
in the tests/ directory — no imports needed.

Why conftest.py?
pytest discovers this file automatically. It's the right place for:
- Shared test data (fake log events, feature dicts)
- Database setup/teardown
- Mock configurations reused across multiple test files
"""

import sys
import os
import pytest
import numpy as np

# Add parent directories to path so tests can import from our services
# In production you'd install packages properly — for a portfolio project,
# path manipulation is the pragmatic solution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'consumer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'producer'))


# ── Shared Test Data ──────────────────────────────────────────────

@pytest.fixture
def normal_log_events():
    """
    A realistic 30-second window of NORMAL log events.
    Low error rate, reasonable latency — should NOT trigger anomaly.
    """
    events = []

    # 38 successful INFO events
    for i in range(38):
        events.append({
            "event_id":    f"normal-{i}",
            "timestamp":   "2024-01-01T00:00:01+00:00",
            "service":     "auth",
            "level":       "INFO",
            "endpoint":    "/login",
            "status_code": 200,
            "latency_ms":  120 + (i % 30),  # slight variation, 120-150ms
            "user_id":     f"user-{i}",
            "ip_address":  "192.168.1.1",
            "message":     "auth processed request successfully in 120ms",
            "fault_injected": False,
        })

    # 2 WARN events — normal background noise
    for i in range(2):
        events.append({
            "event_id":    f"warn-{i}",
            "timestamp":   "2024-01-01T00:00:15+00:00",
            "service":     "auth",
            "level":       "WARN",
            "endpoint":    "/login",
            "status_code": 429,
            "latency_ms":  300,
            "user_id":     f"user-warn-{i}",
            "ip_address":  "192.168.1.2",
            "message":     "auth responded with 429 in 300ms",
            "fault_injected": False,
        })

    return events


@pytest.fixture
def fault_log_events():
    """
    A realistic 30-second window of ANOMALOUS log events.
    High error rate, extreme latency — SHOULD trigger anomaly detection.
    """
    events = []

    # 30 ERROR events with high latency — clear fault signature
    for i in range(30):
        events.append({
            "event_id":    f"fault-{i}",
            "timestamp":   "2024-01-01T00:00:01+00:00",
            "service":     "payment",
            "level":       "ERROR",
            "endpoint":    "/charge",
            "status_code": 503,
            "latency_ms":  3500 + (i * 10),
            "user_id":     f"user-{i}",
            "ip_address":  "10.0.0.1",
            "message":     "payment request failed with status 503 after 3500ms",
            "fault_injected": True,
        })

    # 5 normal events mixed in — real incidents aren't 100% errors
    for i in range(5):
        events.append({
            "event_id":    f"fault-normal-{i}",
            "timestamp":   "2024-01-01T00:00:25+00:00",
            "service":     "payment",
            "level":       "INFO",
            "endpoint":    "/validate",
            "status_code": 200,
            "latency_ms":  200,
            "user_id":     f"user-ok-{i}",
            "ip_address":  "10.0.0.2",
            "message":     "payment processed request successfully in 200ms",
            "fault_injected": True,
        })

    return events


@pytest.fixture
def normal_features():
    """Pre-computed feature dict for a normal window."""
    return {
        "event_count":       40.0,
        "error_rate":        0.02,
        "warn_rate":         0.05,
        "avg_latency_ms":    130.0,
        "p95_latency_ms":    280.0,
        "p99_latency_ms":    310.0,
        "status_5xx_rate":   0.01,
        "status_4xx_rate":   0.03,
        "latency_std":       25.0,
        "error_burst_ratio": 0.1,
        "_error_count":      1,
        "_warn_count":       2,
        "_info_count":       37,
    }


@pytest.fixture
def fault_features():
    """Pre-computed feature dict for an anomalous window."""
    return {
        "event_count":       35.0,
        "error_rate":        0.857,
        "warn_rate":         0.0,
        "avg_latency_ms":    3200.0,
        "p95_latency_ms":    3780.0,
        "p99_latency_ms":    3840.0,
        "status_5xx_rate":   0.857,
        "status_4xx_rate":   0.0,
        "latency_std":       580.0,
        "error_burst_ratio": 0.9,
        "_error_count":      30,
        "_warn_count":       0,
        "_info_count":       5,
    }