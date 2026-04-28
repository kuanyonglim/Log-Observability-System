"""
FastAPI Backend
===============
REST API layer exposing log data and NL query interface.
Endpoints:
  GET  /health       — liveness check
  GET  /logs         — recent log events
  GET  /anomalies    — scored anomaly windows
  POST /query        — natural language → SQL → results
  GET  /metrics      — basic pipeline metrics
  GET  /docs         — auto-generated interactive API docs (free from FastAPI)
"""

import numpy as np
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, text
from sqlalchemy.orm import Session
from sklearn.ensemble import IsolationForest

load_dotenv()

# Local imports — shared models and our new modules
import sys
sys.path.append("/app")

from nl_to_sql import run_nl_query
from schemas import (
    AnomalyAlertResponse,
    LogEventResponse,
    NLQueryRequest,
    NLQueryResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api")

# ── Database Setup ────────────────────────────────────────────────
# We need our SQLAlchemy models — copy them from consumer.
# In production you'd share these via a common package.
# For this project, we duplicate the models file for simplicity.
from sqlalchemy import (
    Boolean, Column, DateTime, Float,
    Integer, String, Text, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

engine         = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base           = declarative_base()


class LogEvent(Base):
    __tablename__ = "log_events"
    id             = Column(Integer, primary_key=True)
    event_id       = Column(String(36), unique=True)
    timestamp      = Column(DateTime(timezone=True), index=True)
    service        = Column(String(50), index=True)
    level          = Column(String(10))
    endpoint       = Column(String(100))
    status_code    = Column(Integer)
    latency_ms     = Column(Integer)
    user_id        = Column(String(36))
    ip_address     = Column(String(45))
    message        = Column(Text)
    fault_injected = Column(Boolean, default=False)
    ingested_at    = Column(DateTime(timezone=True))


class AnomalyAlert(Base):
    __tablename__    = "anomaly_alerts"
    id               = Column(Integer, primary_key=True)
    window_start     = Column(DateTime(timezone=True), index=True)
    window_end       = Column(DateTime(timezone=True))
    service          = Column(String(50), index=True)
    event_count      = Column(Integer)
    error_rate       = Column(Float)
    avg_latency_ms   = Column(Float)
    p95_latency_ms   = Column(Float)
    p99_latency_ms   = Column(Float)
    error_count      = Column(Integer)
    warn_count       = Column(Integer)
    status_5xx_rate  = Column(Float)
    anomaly_score    = Column(Float)
    is_anomaly       = Column(Boolean, default=False)
    explanation      = Column(Text)
    created_at       = Column(DateTime(timezone=True))


# ── App Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE on startup (before serving requests).
    Code after yield runs ONCE on shutdown.

    Why lifespan instead of @app.on_event("startup")?
    lifespan is the modern FastAPI pattern — on_event is deprecated.
    It also makes startup/shutdown logic testable.
    """
    logger.info("🚀 API starting up...")
    # Tables should already exist (created by consumer) but create if not
    Base.metadata.create_all(engine)
    logger.info("✅ Database tables verified")
    yield
    logger.info("👋 API shutting down")


# ── FastAPI App ───────────────────────────────────────────────────
app = FastAPI(
    title="LLM Log Observability API",
    description="Real-time log anomaly detection with natural language querying",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware: allows the Streamlit dashboard (different port)
# to call this API from a browser without being blocked.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependency: DB Session ────────────────────────────────────────
def get_db():
    """
    FastAPI dependency that provides a DB session per request.

    The try/finally ensures the session is always closed,
    even if the endpoint raises an exception.
    yield makes this a 'generator dependency' — FastAPI calls
    next() to get the session, runs your endpoint, then continues
    past yield to run the finally block.
    """
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Liveness probe — used by Docker and load balancers."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/logs", response_model=list[LogEventResponse])
def get_logs(
    # Query parameters with defaults and validation
    limit:   int     = Query(default=100, ge=1, le=1000),
    service: str     | None = Query(default=None),
    level:   str     | None = Query(default=None),
    db:      Session = Depends(get_db),
):
    """
    Fetch recent log events with optional filtering.

    Args:
        limit:   Max rows to return (1-1000)
        service: Filter by service name (auth, payment, api-gateway)
        level:   Filter by log level (INFO, WARN, ERROR)
    """
    query = db.query(LogEvent).order_by(LogEvent.timestamp.desc())

    # Apply filters only if provided — keeps query efficient
    if service:
        query = query.filter(LogEvent.service == service)
    if level:
        query = query.filter(LogEvent.level == level.upper())

    return query.limit(limit).all()


@app.get("/anomalies", response_model=list[AnomalyAlertResponse])
def get_anomalies(
    limit:       int  = Query(default=50, ge=1, le=500),
    service:     str  | None = Query(default=None),
    only_alerts: bool = Query(default=False),
    db:          Session = Depends(get_db),
):
    """
    Fetch anomaly windows with optional filtering.

    Args:
        limit:       Max rows to return
        service:     Filter by service name
        only_alerts: If True, return only windows flagged as anomalous
    """
    query = db.query(AnomalyAlert).order_by(AnomalyAlert.window_start.desc())

    if service:
        query = query.filter(AnomalyAlert.service == service)
    if only_alerts:
        query = query.filter(AnomalyAlert.is_anomaly == True)

    return query.limit(limit).all()


@app.post("/query", response_model=NLQueryResponse)
def natural_language_query(
    body: NLQueryRequest,
    db:   Session = Depends(get_db),
):
    """
    Convert a plain-English question into SQL and return results.

    Example questions:
    - "Show me all ERROR logs from the payment service in the last 5 minutes"
    - "Which service has the highest average latency today?"
    - "How many anomalies were detected in the last hour?"
    - "What's the p99 latency for auth service right now?"
    """
    try:
        result = run_nl_query(
            session=db,
            question=body.question,
            limit=body.limit,
        )
        return result
    except ValueError as e:
        # ValueError = user-facing error (bad question, unsafe SQL)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected error — log it but don't expose internals to client
        logger.error("❌ Unexpected error in /query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    """
    Basic pipeline health metrics.
    Returns aggregate stats useful for the dashboard header.
    """
    try:
        # Total log events ingested
        total_logs = db.execute(
            text("SELECT COUNT(*) FROM log_events")
        ).scalar()

        # Total anomaly windows scored
        total_windows = db.execute(
            text("SELECT COUNT(*) FROM anomaly_alerts")
        ).scalar()

        # Total anomalies detected
        total_anomalies = db.execute(
            text("SELECT COUNT(*) FROM anomaly_alerts WHERE is_anomaly = true")
        ).scalar()

        # Logs per service breakdown
        service_counts = db.execute(text("""
            SELECT service, COUNT(*) as count
            FROM log_events
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            GROUP BY service
            ORDER BY count DESC
        """)).fetchall()

        # Most recent anomaly
        last_anomaly = db.execute(text("""
            SELECT service, window_start, anomaly_score
            FROM anomaly_alerts
            WHERE is_anomaly = true
            ORDER BY window_start DESC
            LIMIT 1
        """)).fetchone()

        return {
            "total_logs_ingested":     total_logs,
            "total_windows_scored":    total_windows,
            "total_anomalies_flagged": total_anomalies,
            "anomaly_rate_pct": round(
                (total_anomalies / total_windows * 100) if total_windows else 0, 2
            ),
            "logs_last_hour_by_service": [
                {"service": row[0], "count": row[1]} for row in service_counts
            ],
            "last_anomaly": {
                "service":      last_anomaly[0],
                "window_start": last_anomaly[1].isoformat(),
                "score":        last_anomaly[2],
            } if last_anomaly else None,
        }
    except Exception as e:
        logger.error("❌ Metrics error: %s", e)
        raise HTTPException(status_code=500, detail="Could not fetch metrics")
    
@app.post("/recalibrate")
def recalibrate_model(
    contamination: float = Query(
        default=0.05,
        ge=0.01,
        le=0.5,
        description="Expected fraction of anomalies (0.01–0.5)"
    ),
    db: Session = Depends(get_db),
):
    """
    Retrain Isolation Forest on real accumulated window data
    and reclassify all historical windows with the new model.

    Why this is useful for demos:
    The consumer starts with synthetic bootstrap data. After real
    traffic accumulates, this endpoint retrains on actual patterns —
    dramatically reducing false positives from the bootstrap phase.

    Steps:
    1. Fetch all scored windows from anomaly_alerts
    2. Retrain Isolation Forest on their feature vectors
    3. Reclassify every window with the new model
    4. Update is_anomaly in the DB so dashboard reflects it immediately
    """
    try:
        # ── Step 1: Fetch all window feature vectors ──────────────
        rows = db.execute(text("""
            SELECT
                id,
                event_count,
                error_rate,
                COALESCE(
                    (SELECT AVG(latency_ms) FROM log_events l
                     WHERE l.service = a.service
                     AND l.timestamp BETWEEN a.window_start AND a.window_end),
                    avg_latency_ms
                ) as warn_rate,
                avg_latency_ms,
                p95_latency_ms,
                p99_latency_ms,
                status_5xx_rate,
                COALESCE(
                    CAST(error_count AS FLOAT) / NULLIF(event_count, 0),
                    0
                ) as status_4xx_rate,
                anomaly_score
            FROM anomaly_alerts a
            ORDER BY window_start ASC
        """)).fetchall()

        if len(rows) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough data to recalibrate — need at least 10 windows, "
                       f"got {len(rows)}. Wait a few minutes for more data to accumulate."
            )

        # ── Step 2: Build feature matrix ──────────────────────────
        # Extract the 6 most reliable features we have stored
        # (we don't store all 10 original features, so we use what's available)
        feature_matrix = []
        window_ids     = []

        for row in rows:
            features = [
                float(row[1] or 0),   # event_count
                float(row[2] or 0),   # error_rate
                float(row[4] or 0),   # avg_latency_ms
                float(row[5] or 0),   # p95_latency_ms
                float(row[6] or 0),   # p99_latency_ms
                float(row[7] or 0),   # status_5xx_rate
            ]
            feature_matrix.append(features)
            window_ids.append(row[0])  # id column

        X = np.array(feature_matrix)

        # ── Step 3: Retrain Isolation Forest ──────────────────────
        logger.info(
            "🔄 Recalibrating Isolation Forest on %d real windows "
            "with contamination=%.2f", len(rows), contamination
        )

        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X)

        # ── Step 4: Reclassify all windows ────────────────────────
        new_scores      = model.score_samples(X)
        new_predictions = model.predict(X)   # 1 = normal, -1 = anomaly

        updated_anomalies = 0
        updated_normal    = 0

        for i, window_id in enumerate(window_ids):
            new_score     = float(new_scores[i])
            new_is_anomaly = bool(new_predictions[i] == -1)

            db.execute(text("""
                UPDATE anomaly_alerts
                SET anomaly_score = :score,
                    is_anomaly    = :is_anomaly
                WHERE id = :id
            """), {
                "score":      new_score,
                "is_anomaly": new_is_anomaly,
                "id":         window_id,
            })

            if new_is_anomaly:
                updated_anomalies += 1
            else:
                updated_normal += 1

        db.commit()

        anomaly_rate = (updated_anomalies / len(window_ids)) * 100
        logger.info(
            "✅ Recalibration complete. %d anomalies / %d normal (%.1f%%)",
            updated_anomalies, updated_normal, anomaly_rate
        )

        return {
            "status":            "recalibrated",
            "windows_analysed":  len(window_ids),
            "contamination_used": contamination,
            "anomalies_flagged": updated_anomalies,
            "normal_windows":    updated_normal,
            "anomaly_rate_pct":  round(anomaly_rate, 2),
            "message": (
                f"Model retrained on {len(window_ids)} real windows. "
                f"{updated_anomalies} anomalies detected ({anomaly_rate:.1f}%). "
                f"Dashboard will reflect changes on next refresh."
            )
        }

    except HTTPException:
        raise  # re-raise our own HTTP exceptions unchanged
    except Exception as e:
        db.rollback()
        logger.error("❌ Recalibration failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recalibration failed: {str(e)}")