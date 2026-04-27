"""
Pydantic Schemas
================
Define the shape of every API request and response.

Why separate schemas from DB models?
Your DB model (SQLAlchemy) defines HOW data is stored.
Your schema (Pydantic) defines HOW data is exposed via API.
They're often similar but not identical — for example, you might
expose anomaly_score but not the internal _error_count field.
Keeping them separate means you can change one without breaking the other.
"""

from datetime import datetime
from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


# ── Response Schemas ──────────────────────────────────────────────

class LogEventResponse(BaseModel):
    """Shape of a single log event returned by GET /logs"""

    # model_config tells Pydantic to read attributes from ORM objects
    # (SQLAlchemy rows) directly, not just from dicts.
    # Without this, FastAPI can't serialize SQLAlchemy model instances.
    model_config = ConfigDict(from_attributes=True)

    id:            int
    event_id:      str
    timestamp:     datetime
    service:       str
    level:         str
    endpoint:      str | None
    status_code:   int | None
    latency_ms:    int | None
    message:       str | None
    fault_injected: bool
    ingested_at:   datetime


class AnomalyAlertResponse(BaseModel):
    """Shape of a single anomaly window returned by GET /anomalies"""
    model_config = ConfigDict(from_attributes=True)

    id:              int
    window_start:    datetime
    window_end:      datetime
    service:         str
    event_count:     int    | None
    error_rate:      float  | None
    avg_latency_ms:  float  | None
    p95_latency_ms:  float  | None
    p99_latency_ms:  float  | None
    error_count:     int    | None
    warn_count:      int    | None
    status_5xx_rate: float  | None
    anomaly_score:   float  | None
    is_anomaly:      bool
    explanation:     str    | None
    created_at:      datetime


# ── Request Schemas ───────────────────────────────────────────────

class NLQueryRequest(BaseModel):
    # Annotated lets us attach constraints to a type.
    # StringConstraints is the Pydantic v2 way to add string-specific rules.
    # strip_whitespace=True: "   " → "" before min_length is checked.
    # min_length=1: after stripping, empty string fails → 422 response.
    question: Annotated[str, StringConstraints(
        strip_whitespace=True,
        min_length=1
    )]
    limit: int = Field(default=100, ge=1, le=500)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Show me all ERROR logs from the payment service",
                "limit": 50
            }
        }
    )


class NLQueryResponse(BaseModel):
    """Response from POST /query"""
    question:      str        # echo back the original question
    generated_sql: str        # the SQL Claude produced — transparency matters
    results:       list[dict[str, Any]]  # the actual query results
    row_count:     int
    explanation:   str        # Claude's plain-English summary of results