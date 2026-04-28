"""
NL-to-SQL Engine
================
Converts plain-English questions into safe, executable PostgreSQL queries
using Google Gemini as the reasoning engine (free tier).

The pipeline:
  1. Build a system prompt with full schema context
  2. Send user question to Gemini
  3. Extract and validate the generated SQL
  4. Execute against PostgreSQL
  5. Ask Gemini to summarise the results in plain English
"""

import logging
import os
import re

import google.generativeai as genai
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("nl_to_sql")

# ── Gemini Client ─────────────────────────────────────────────────
_model: genai.GenerativeModel | None = None

def get_client() -> genai.GenerativeModel:
    """
    Lazy singleton for the Gemini client.
    Reads GEMINI_API_KEY from environment on first call.
    """
    # Change between these models for test purposes: [model_name] | [requests]
    # "gemini-2.5-flash"                | 5 tokens
    # "gemini-3-flash-preview"          | 20 tokens (unstable)
    # "gemini-3.1-flash-lite-preview"   | 15 tokens
    global _model
    if _model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to your .env file. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(
            model_name="gemini-3.1-flash-lite-preview",
            generation_config=genai.GenerationConfig(
                temperature=0,        # deterministic SQL generation
                max_output_tokens=500,
            )
        )
        logger.info("✅ Gemini client initialised")
    return _model


# ── Schema Injection ──────────────────────────────────────────────
SCHEMA_CONTEXT = """
You have access to a PostgreSQL database with the following schema:

TABLE: log_events
  id              INTEGER PRIMARY KEY
  event_id        VARCHAR(36) UNIQUE
  timestamp       TIMESTAMPTZ          -- when the log was generated (UTC)
  service         VARCHAR(50)          -- one of: 'auth', 'payment', 'api-gateway'
  level           VARCHAR(10)          -- one of: 'INFO', 'WARN', 'ERROR'
  endpoint        VARCHAR(100)         -- HTTP endpoint e.g. '/login', '/charge'
  status_code     INTEGER              -- HTTP status code e.g. 200, 404, 503
  latency_ms      INTEGER              -- request latency in milliseconds
  user_id         VARCHAR(36)          -- UUID of the requesting user
  ip_address      VARCHAR(45)          -- client IP address
  message         TEXT                 -- human-readable log message
  fault_injected  BOOLEAN              -- true if this was a synthetic fault event
  ingested_at     TIMESTAMPTZ          -- when the pipeline consumed this log

TABLE: anomaly_alerts
  id              INTEGER PRIMARY KEY
  window_start    TIMESTAMPTZ          -- start of the 30-second analysis window
  window_end      TIMESTAMPTZ          -- end of the 30-second analysis window
  service         VARCHAR(50)          -- which service this window covers
  event_count     INTEGER              -- total log events in this window
  error_rate      FLOAT                -- fraction of events that were ERROR/WARN (0.0-1.0)
  avg_latency_ms  FLOAT                -- mean latency across all events in window
  p95_latency_ms  FLOAT                -- 95th percentile latency
  p99_latency_ms  FLOAT                -- 99th percentile latency
  error_count     INTEGER              -- raw count of ERROR-level events
  warn_count      INTEGER              -- raw count of WARN-level events
  status_5xx_rate FLOAT                -- fraction of 5xx HTTP responses
  anomaly_score   FLOAT                -- Isolation Forest score (lower = more anomalous)
  is_anomaly      BOOLEAN              -- true if this window was flagged as anomalous
  explanation     TEXT                 -- LLM-generated explanation (may be null)
  created_at      TIMESTAMPTZ          -- when this alert was recorded

IMPORTANT RULES you must follow exactly:
- Generate ONLY a single SELECT statement. Never INSERT, UPDATE, DELETE, DROP, or ALTER.
- Always use LIMIT {limit} unless the question asks for aggregates (COUNT, AVG, etc.)
- For time-based questions like 'last 10 minutes', use: timestamp > NOW() - INTERVAL '10 minutes'
- For 'recent' without a time specified, default to: timestamp > NOW() - INTERVAL '1 hour'
- String comparisons on 'service' and 'level' are case-sensitive — use exact values
- Return ONLY the raw SQL query — no explanation, no markdown, no code fences, no backticks
- If the question cannot be answered with these tables, return exactly: CANNOT_ANSWER
"""

SUMMARY_PROMPT = """The user asked: "{question}"

The SQL query executed was:
{sql}

The results were ({total} total rows, showing first {shown}):
{rows}

Provide a concise 2-3 sentence plain-English summary of what these results show.
Focus on key insights or patterns. Do not repeat the SQL."""


# ── SQL Safety Validator ──────────────────────────────────────────
_ALLOWED_SQL_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE | re.MULTILINE)
_DANGEROUS_KEYWORDS  = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE"
    r"|GRANT|REVOKE|COPY|pg_read_file|pg_ls_dir)\b",
    re.IGNORECASE
)
_COMMENT_PATTERN = re.compile(r"(--|/\*|\*/)")


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate that generated SQL is safe to execute.
    Returns (is_valid, error_message).
    """
    sql = sql.strip()

    if sql == "CANNOT_ANSWER":
        return False, "CANNOT_ANSWER"

    # Strip markdown fences if Gemini adds them despite instructions
    # e.g. ```sql SELECT ... ``` → SELECT ...
    if sql.startswith("```"):
        lines = sql.split("\n")
        # Remove first line (```sql or ```) and last line (```)
        sql = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    if not _ALLOWED_SQL_PATTERN.match(sql):
        logger.warning("SQL rejected — does not start with SELECT: %s", sql[:100])
        return False, "Query must be a SELECT statement"

    match = _DANGEROUS_KEYWORDS.search(sql)
    if match:
        logger.warning("SQL rejected — dangerous keyword: %s", match.group())
        return False, f"Query contains forbidden keyword: {match.group()}"

    if _COMMENT_PATTERN.search(sql):
        logger.warning("SQL rejected — contains SQL comments")
        return False, "Query must not contain SQL comments"

    if len(sql) > 500:
        return False, "Query too long"

    return True, sql  # return cleaned sql as second element when valid


# ── Gemini Calls ──────────────────────────────────────────────────
def generate_sql(question: str, limit: int = 100) -> str:
    """Ask Gemini to convert a natural language question into SQL."""
    logger.info("🤖 Generating SQL for: '%s'", question)

    prompt = SCHEMA_CONTEXT.format(limit=limit) + f"\n\nUser question: {question}"

    response = get_client().generate_content(prompt)
    sql      = response.text.strip()

    # Strip markdown fences Gemini sometimes adds
    if sql.startswith("```"):
        lines = [l for l in sql.split("\n") if not l.strip().startswith("```")]
        sql   = "\n".join(lines).strip()

    logger.info("📝 Generated SQL: %s", sql)
    return sql


def execute_sql(session: Session, sql: str, limit: int = 100) -> list[dict]:
    """Execute a validated SQL query and return results as list of dicts."""
    if "LIMIT" not in sql.upper():
        sql = f"{sql.rstrip(';')} LIMIT {limit}"

    result  = session.execute(text(sql))
    columns = list(result.keys())
    rows    = [dict(zip(columns, row)) for row in result.fetchall()]

    logger.info("✅ Query returned %d rows", len(rows))
    return rows


def summarise_results(question: str, sql: str, rows: list[dict]) -> str:
    """Ask Gemini to summarise query results in plain English."""
    if not rows:
        return "No results found for your query."

    sample = rows[:20]
    prompt = SUMMARY_PROMPT.format(
        question=question,
        sql=sql,
        total=len(rows),
        shown=len(sample),
        rows=sample,
    )

    summary_model = genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite-preview",
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=300,
        )
    )
    response = summary_model.generate_content(prompt)
    return response.text.strip()


# ── Main Pipeline ─────────────────────────────────────────────────
def run_nl_query(session: Session, question: str, limit: int = 100) -> dict:
    """
    Full NL-to-SQL pipeline. Orchestrates all steps.
    Raises ValueError with user-friendly message on failure.
    """
    # Step 1: Generate SQL
    raw_sql = generate_sql(question, limit)

    # Step 2: Validate SQL — validate_sql returns cleaned sql as second
    # element when valid, so we capture it
    is_valid, result = validate_sql(raw_sql)

    if not is_valid:
        if result == "CANNOT_ANSWER":
            raise ValueError(
                "This question cannot be answered with the available data. "
                "Try asking about log levels, latency, error rates, or anomalies."
            )
        raise ValueError(f"Generated SQL failed safety validation: {result}")

    # Use the cleaned SQL (markdown stripped) for execution
    clean_sql = result

    # Step 3: Execute
    try:
        rows = execute_sql(session, clean_sql, limit)
    except Exception as e:
        logger.error("SQL execution failed: %s\nSQL: %s", e, clean_sql)
        raise ValueError(f"Query execution failed: {str(e)}")

    # Step 4: Summarise
    explanation = summarise_results(question, clean_sql, rows)

    return {
        "question":      question,
        "generated_sql": clean_sql,
        "results":       rows,
        "row_count":     len(rows),
        "explanation":   explanation,
    }