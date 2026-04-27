"""
Tests for log event generation logic.

We don't test Kafka connectivity here (that's an integration test).
We focus on the data quality of generated events — the contract
downstream consumers depend on.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'producer'))

from main import (
    generate_log_event,
    FaultScheduler,
    SERVICE_PROFILES,
    SERVICES,
)


class TestGenerateLogEvent:

    def test_returns_required_fields(self):
        """
        Every log event must have these fields.
        The consumer's feature extractor and DB schema depend on them.
        """
        required = [
            "event_id", "timestamp", "service", "level",
            "endpoint", "status_code", "latency_ms",
            "user_id", "ip_address", "message", "fault_injected",
        ]
        event = generate_log_event("auth", fault_mode=False)
        for field in required:
            assert field in event, f"Missing required field: {field}"

    def test_service_name_preserved(self):
        """Generated event must carry the service name it was called with."""
        for service in SERVICES:
            event = generate_log_event(service)
            assert event["service"] == service

    def test_valid_log_level(self):
        """Level must be one of the three valid values."""
        valid_levels = {"INFO", "WARN", "ERROR"}
        for _ in range(50):
            event = generate_log_event("auth")
            assert event["level"] in valid_levels, (
                f"Invalid level: {event['level']}"
            )

    def test_latency_is_positive(self):
        """Latency can never be negative — would be meaningless."""
        for _ in range(50):
            event = generate_log_event("payment")
            assert event["latency_ms"] > 0, (
                f"Latency must be positive, got {event['latency_ms']}"
            )

    def test_fault_mode_increases_latency(self):
        """
        Fault events must have higher average latency than normal events.
        This is the fundamental property our ML model detects.
        Run 200 samples to get a stable average.
        """
        normal_latencies = [
            generate_log_event("payment", fault_mode=False)["latency_ms"]
            for _ in range(200)
        ]
        fault_latencies = [
            generate_log_event("payment", fault_mode=True)["latency_ms"]
            for _ in range(200)
        ]
        normal_avg = sum(normal_latencies) / len(normal_latencies)
        fault_avg  = sum(fault_latencies)  / len(fault_latencies)

        assert fault_avg > normal_avg * 2, (
            f"Fault latency ({fault_avg:.0f}ms) should be >2x "
            f"normal ({normal_avg:.0f}ms)"
        )

    def test_fault_mode_increases_error_rate(self):
        """Fault mode must produce more ERROR/WARN events than normal mode."""
        normal_errors = sum(
            1 for _ in range(200)
            if generate_log_event("auth", fault_mode=False)["level"] in ("ERROR", "WARN")
        )
        fault_errors = sum(
            1 for _ in range(200)
            if generate_log_event("auth", fault_mode=True)["level"] in ("ERROR", "WARN")
        )
        assert fault_errors > normal_errors, (
            f"Fault mode ({fault_errors} errors) should produce more errors "
            f"than normal mode ({normal_errors} errors)"
        )

    def test_fault_injected_flag_set_correctly(self):
        """fault_injected field must reflect the mode the event was generated in."""
        normal_event = generate_log_event("auth", fault_mode=False)
        fault_event  = generate_log_event("auth", fault_mode=True)
        assert normal_event["fault_injected"] is False
        assert fault_event["fault_injected"]  is True

    def test_event_id_is_unique(self):
        """Every event must have a unique ID — duplicates corrupt the DB."""
        ids = {generate_log_event("auth")["event_id"] for _ in range(100)}
        assert len(ids) == 100, "All 100 event IDs must be unique"

    def test_endpoint_belongs_to_service(self):
        """Endpoints must come from the service's own profile."""
        for service in SERVICES:
            valid_endpoints = SERVICE_PROFILES[service]["endpoints"]
            for _ in range(20):
                event = generate_log_event(service)
                assert event["endpoint"] in valid_endpoints, (
                    f"Endpoint {event['endpoint']} not valid for {service}"
                )


class TestFaultScheduler:

    def test_no_fault_initially(self):
        """
        Scheduler should not be in fault mode at startup.
        We set a very long interval to ensure no fault triggers immediately.
        """
        scheduler = FaultScheduler(interval=9999, duration=10)
        is_fault, service = scheduler.tick()
        assert is_fault is False
        assert service is None

    def test_fault_activates_after_interval(self):
        """After the interval elapses, fault mode must activate."""
        import time
        # interval=0 means fault triggers immediately on first tick
        scheduler = FaultScheduler(interval=0, duration=10)
        time.sleep(0.01)  # tiny sleep to ensure interval has passed
        is_fault, service = scheduler.tick()
        assert is_fault is True
        assert service in SERVICES

    def test_fault_affects_valid_service(self):
        """The faulted service must be one of our 3 known services."""
        import time
        scheduler = FaultScheduler(interval=0, duration=10)
        time.sleep(0.01)
        _, service = scheduler.tick()
        if service is not None:
            assert service in SERVICES