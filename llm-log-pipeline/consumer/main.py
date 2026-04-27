"""
Kafka Consumer + Anomaly Pipeline
==================================
Consumes log events from Kafka, builds 30-second tumbling windows
per service, extracts features, scores anomalies, persists to PostgreSQL.
"""

import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

from dotenv import load_dotenv
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from features import extract_features
from models import AnomalyAlert, LogEvent, get_engine, get_session_factory, init_db
from scorer import AnomalyScorer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("consumer")

# ── Config ────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "app-logs")
DATABASE_URL            = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)
WINDOW_SIZE_SECONDS     = int(os.getenv("WINDOW_SIZE_SECONDS", "30"))
ANOMALY_CONTAMINATION   = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))


# ── Kafka Connection ──────────────────────────────────────────────
def create_consumer(retries: int = 15, delay: int = 5) -> KafkaConsumer:
    """Connect to Kafka with retries. Same pattern as producer."""
    import json

    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                # Deserialise incoming bytes back to Python dicts
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                # consumer_timeout_ms: how long to block waiting for messages.
                # 1000ms = poll for 1s then yield control back to our loop.
                # Without this, the for loop blocks forever on empty topics.
                consumer_timeout_ms=1000,
                # auto_offset_reset='earliest': if this consumer hasn't read
                # this topic before, start from the very first message.
                # 'latest' would skip all existing messages.
                auto_offset_reset="earliest",
                # group_id: identifies this consumer in a consumer group.
                # Kafka tracks which messages each group has processed.
                # If the consumer restarts, it resumes from where it left off.
                group_id="anomaly-detector",
            )
            logger.info("✅ Connected to Kafka topic '%s'", KAFKA_TOPIC)
            return consumer
        except NoBrokersAvailable:
            logger.warning(
                "⏳ Kafka not ready (attempt %d/%d). Retrying in %ds...",
                attempt, retries, delay
            )
            time.sleep(delay)

    raise RuntimeError("❌ Could not connect to Kafka")


# ── Window Manager ────────────────────────────────────────────────
class WindowManager:
    """
    Manages 30-second tumbling windows per service.

    Design: one independent window per service. Why per-service?
    A payment service spike shouldn't pollute the auth service's window.
    Each service has its own normal baseline.

    Structure:
        _windows[service] = {
            "events":     [...],   # events accumulated in this window
            "start_time": float,   # Unix timestamp when window opened
        }
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        # defaultdict auto-creates missing keys — no KeyError on first access
        self._windows: dict[str, dict] = defaultdict(
            lambda: {"events": [], "start_time": time.time()}
        )

    def add_event(self, event: dict) -> list[tuple[str, dict, float, float]]:
        """
        Add an event to its service's window.

        Returns a list of (service, events, window_start, window_end) tuples
        for any windows that have just closed. Usually returns [].
        Occasionally returns one tuple when a 30s window completes.
        """
        service = event["service"]
        window  = self._windows[service]
        window["events"].append(event)

        closed_windows = []
        now = time.time()

        # Has this window exceeded our time limit?
        if now - window["start_time"] >= self.window_size:
            closed_windows.append((
                service,
                window["events"].copy(),
                window["start_time"],
                now,
            ))
            # Reset the window for this service
            self._windows[service] = {
                "events":     [],
                "start_time": now,
            }

        return closed_windows


# ── Database Writer ───────────────────────────────────────────────
def write_log_event(session, event: dict) -> None:
    """Persist a single log event to PostgreSQL."""
    row = LogEvent(
        event_id    = event["event_id"],
        timestamp   = datetime.fromisoformat(
            event["timestamp"].replace("Z", "+00:00")
        ),
        service     = event["service"],
        level       = event["level"],
        endpoint    = event["endpoint"],
        status_code = event["status_code"],
        latency_ms  = event["latency_ms"],
        user_id     = event["user_id"],
        ip_address  = event["ip_address"],
        message     = event["message"],
        fault_injected = event.get("fault_injected", False),
    )
    # merge() instead of add(): handles duplicates gracefully.
    # If we restart and replay Kafka messages, event_ids we've already
    # seen won't cause unique constraint violations.
    session.merge(row)


def write_anomaly_alert(
    session,
    service:      str,
    features:     dict,
    score:        float,
    is_anomaly:   bool,
    window_start: float,
    window_end:   float,
) -> AnomalyAlert:
    """Persist a scored window to the anomaly_alerts table."""
    alert = AnomalyAlert(
        window_start    = datetime.fromtimestamp(window_start, tz=timezone.utc),
        window_end      = datetime.fromtimestamp(window_end,   tz=timezone.utc),
        service         = service,
        event_count     = int(features["event_count"]),
        error_rate      = features["error_rate"],
        avg_latency_ms  = features["avg_latency_ms"],
        p95_latency_ms  = features["p95_latency_ms"],
        p99_latency_ms  = features["p99_latency_ms"],
        error_count     = features["_error_count"],
        warn_count      = features["_warn_count"],
        status_5xx_rate = features["status_5xx_rate"],
        anomaly_score   = score,
        is_anomaly      = is_anomaly,
    )
    session.add(alert)
    return alert


# ── Main Loop ─────────────────────────────────────────────────────
def main() -> None:
    logger.info("Waiting 20s for infrastructure to stabilise...")
    time.sleep(20)

    # ── Setup DB ─────────────────────────────────────────────────
    engine          = get_engine(DATABASE_URL)
    SessionFactory  = get_session_factory(engine)
    init_db(engine)  # CREATE TABLE IF NOT EXISTS
    logger.info("✅ Database initialised")

    # ── Setup ML model ───────────────────────────────────────────
    scorer  = AnomalyScorer(contamination=ANOMALY_CONTAMINATION)

    # ── Setup Kafka consumer ──────────────────────────────────────
    consumer = create_consumer()
    windows  = WindowManager(window_size=WINDOW_SIZE_SECONDS)

    logger.info("🚀 Consumer started. Listening for log events...")

    events_processed = 0
    anomalies_found  = 0

    while True:
        # ── Consume a batch of messages from Kafka ────────────────
        # The for loop runs until consumer_timeout_ms (1s) with no messages,
        # then exits — giving us a chance to check window timeouts.
        session = SessionFactory()
        try:
            for message in consumer:
                event = message.value  # already deserialised to dict

                # Persist raw log to PostgreSQL
                write_log_event(session, event)
                events_processed += 1

                # Add to rolling window; get back any windows that just closed
                closed = windows.add_event(event)

                for service, win_events, win_start, win_end in closed:
                    # ── Feature extraction ────────────────────────
                    features = extract_features(win_events)

                    # ── Anomaly scoring ───────────────────────────
                    score, is_anomaly = scorer.score(features)

                    # ── Persist the scored window ─────────────────
                    alert = write_anomaly_alert(
                        session, service, features,
                        score, is_anomaly, win_start, win_end
                    )

                    if is_anomaly:
                        anomalies_found += 1
                        logger.warning(
                            "🚨 ANOMALY DETECTED | service=%s score=%.4f "
                            "error_rate=%.1f%% p99=%dms events=%d",
                            service, score,
                            features["error_rate"] * 100,
                            features["p99_latency_ms"],
                            features["event_count"],
                        )
                    else:
                        logger.debug(
                            "✅ Normal window | service=%s score=%.4f events=%d",
                            service, score, features["event_count"]
                        )

                # Commit every 50 events to balance write frequency vs
                # transaction overhead. Committing every event = 50x more
                # round-trips to Postgres. Batching helps throughput.
                if events_processed % 50 == 0:
                    session.commit()
                    logger.info(
                        "📊 Processed: %d events | Anomalies: %d",
                        events_processed, anomalies_found
                    )

            # Commit any remaining unflushed events after timeout
            session.commit()

        except Exception as e:
            logger.error("❌ Error in consumer loop: %s", e, exc_info=True)
            session.rollback()
        finally:
            session.close()


if __name__ == "__main__":
    main()