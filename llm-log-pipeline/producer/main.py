"""
Log Producer
============
Simulates 3 microservices generating structured JSON log events.
Publishes events to a Kafka topic at configurable rates.
Injects fault bursts periodically to create detectable anomalies in the logs.
"""

import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from faker import Faker
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# ── Bootstrap ────────────────────────────────────────────────────
load_dotenv() # pulls variables from .env into os.environ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("producer")

fake = Faker() # Initialize Faker

# ── Constants ────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "app-logs")

# Microservices
SERVICES = ["auth", "payment", "api-gateway"]

# Each service has its own typical behavior profile
# (base_error_rate, base_latency_ms_mean, base_latency_ms_std)
SERVICE_PROFILES: dict[str, dict] = {
    "auth": {
        "base_error_rate": 0.02,    # 2% errors normally
        "latency_mean":    120,     # ~120ms average
        "latency_std":     30,
        "endpoints": ["/login", "/logout", "/refresh", "/register"]
    },
    "payment": {
        "base_error_rate": 0.01,    # 1% errors - payments should be reliable
        "latency_mean":    250,     # payments are slower (DB + external calls)
        "latency_std":     60,
        "endpoints": ["/charge", "/refund", "/validate", "/history"]
    },
    "api-gateway": {
        "base_error_rate": 0.03,   
        "latency_mean":    80,
        "latency_std":     20,
        "endpoints": ["/health", "/route", "/auth-check", "/rate-limit"],
    },
}

# HTTP status codes grouped by meaning - used to build realistic responses
STATUS_CODES = {
    "success": [200, 201, 204],
    "client_error": [400, 401, 403, 404, 429],
    "server_error": [500, 502, 503, 504],
}

# ── Kafka Connection ──────────────────────────────────────────────
def create_producer(retries: int = 10, delay: int = 5) -> KafkaProducer:
    """
    Attempt to connect to Kafka with retries.

    Why retries? Docker starts containers in parallel. Kafka might not be
    ready when the producer container starts, even with healthchecks.
    Retrying with a delay is the standard resilience pattern.
    """
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                # Serialize every message to JSON bytes before sending.
                # Kafka only understands bytes - not Python dicts.
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Wait for the partition leader to confirm receipt.
                # acks=1 balances durability with speed (vs acks=0 or acks='all')
                acks=1,
                # Retry failed sends up to 3 times automatically
                retries=3,
            )
            logger.info("✅ Connected to Kafka at %s", KAFKA_BOOTSTRAP_SERVERS)
            return producer
        except NoBrokersAvailable:
            logger.warning(
                "⏳ Kafka not ready (attempt %d/%d). Retrying in %ds...",
                attempt, retries, delay
            )
            time.sleep(delay)
        
    raise RuntimeError("❌ Could not connect to Kafka after %d attempts" % retries)

# ── Log Event Generation ──────────────────────────────────────────
def generate_log_event(service: str, fault_mode: bool = False) -> dict:
    """
    Generate a single structured log event for a given service.

    Args:
        service:    One of 'auth', 'payment', 'api-gateway'
        fault_mode: If True, inject elevated errors and high latency 
                    to simulate an incident

    Returns:
        A dict representing one log line - will be serialized to JSON
    """
    profile = SERVICE_PROFILES[service]

    # ── Determine error rate for this event ──────────────────────
    # In fault mode we spike the error rate dramatically (up to 60%)
    # This is what Isolation Forest will detect as anomalous
    error_rate = profile["base_error_rate"] * 15 if fault_mode else profile["base_error_rate"]
    error_rate = min(error_rate, 0.60)  # cap at 60% so logs aren't all errors

    # ── Decide outcome: error or success? ────────────────────────
    is_error = random.random() < error_rate

    # ── Simulate latency ─────────────────────────────────────────
    # Normal distribution around the service's typical latency.
    # In fault mode: latency triples (timeouts, resource exhaustion).
    latency_multiplier = random.uniform(2.5, 4.0) if fault_mode else 1.0
    latency_ms = max(
        1,  # latency cannot be negative
        int(random.gauss(
            profile["latency_mean"] * latency_multiplier,
            profile["latency_std"] * latency_multiplier
        ))        
    )

    # ── Choose log level ─────────────────────────────────────────
    if is_error and latency_ms > 1000:
        level = "ERROR"
    elif is_error:
        level = "WARN"
    else:
        level = "INFO"

    # ── Choose HTTP status code ───────────────────────────────────
    if is_error:
        # Weight: 70% server errors, 30% client errors during faults
        # (servers crashing is more interesting than bad requests)
        status_code = random.choice(
            STATUS_CODES["server_error"] * 7 + STATUS_CODES["client_error"] * 3
            if fault_mode
            else STATUS_CODES["server_error"] + STATUS_CODES["client_error"]
        )
    else:
        status_code = random.choice(STATUS_CODES["success"])

    # ── Compose the structured log event ─────────────────────────
    return {
        "event_id":         str(uuid.uuid4()),      # unique ID per log line
        "timestamp":        datetime.now(timezone.utc).isoformat(), # time should be utc for same formatting
        "service":          service,
        "level":            level,
        "endpoint":         random.choice(profile["endpoints"]),
        "status_code":      status_code,
        "latency_ms":       latency_ms,
        "user_id":          fake.uuid4(),           # realistic fake user ID
        "ip_address":       fake.ipv4_public(),     # realistic fake IP
        "message":          _build_message(service, level, status_code, latency_ms),
        "fault_injected":   fault_mode,           # label for debugging/evaluation
    }

def _build_message(service: str, level: str, status_code: int, latency_ms: int) -> str:
    """ Build a human-readable log message string from structured fields. """
    if level == "ERROR":
        return f"{service} request failed with status {status_code} after {latency_ms}ms"
    elif level == "WARN":
        return f"{service} responded with {status_code} in {latency_ms}ms - elevated latency"
    else:
        return f"{service} processed request successfully in {latency_ms}ms"
    

# ── Fault Injection Scheduler ─────────────────────────────────────
class FaultScheduler:
    """
    Manages  periodic fault bursts.

    Design: every 'interval' seconds, trigger a fault window lasting 
    'duration' seconds. During this window, one random service is 
    affected. This creates the anomaly spike our ML model must detect.
    """

    def __init__(self, interval: int = 120, duration: int = 20):
        self.interval = interval    # seconds between fault bursts
        self.duration = duration    # how long each burst lasts
        self._fault_until: float = 0.0
        self._next_fault: float  = time.time() + interval
        self._affected_service: str | None = None

    def tick(self) -> tuple[bool, str | None]:
        """
        Call once per loop iteration:

        Returns:
            (is_fault_active, affected_service_name)
        """
        now = time.time()

        # Time to trigger a new fault?
        if now >= self._next_fault and now > self._fault_until:
            self._affected_service  = random.choice(SERVICES)
            self._fault_until       = now + self.duration
            self._next_fault        = now + self.interval
            logger.warning(
                "🔴 FAULT INJECTED on '%s' for %ds",
                self._affected_service, self.duration
            )

        # Is a fault currently active?
        if now < self._fault_until:
            return True, self._affected_service
        
        return False, None

# ── Main Loop ─────────────────────────────────────────────────────
def main() -> None:
    producer    = create_producer()
    fault_sched = FaultScheduler(interval=120, duration=20)
    events_sent = 0

    logger.info("🚀 Log producer started. Publishing to topic '%s'", KAFKA_TOPIC)

    while True:
        fault_active, fault_service = fault_sched.tick()
        
        for service in SERVICES:
            # Only the affected service runs in fault mode
            in_fault = fault_active and (service == fault_service)

            # Generate between 1-5 log events per service per loop tick.
            # This simulates variable traffic - some services are busier
            batch_size = random.randint(3, 8) if in_fault else random.randint(1, 4)

            for _ in range(batch_size):
                event = generate_log_event(service, fault_mode=in_fault)

                # Send to Kafka - the value_serializer handles JSON encoding
                producer.send(KAFKA_TOPIC, value=event)
                events_sent += 1

        # Flush ensures buffered messages are actually sent to Kafka
        # Without flush, messages can sit in a bufer and be lost on crash
        producer.flush()

        if events_sent % 100 == 0:
            logger.info("📨 Total events published: %d", events_sent)

        # Sleep between bursts - adjust to control log throughout
        time.sleep(1)

if __name__ == "__main__":
    main()







