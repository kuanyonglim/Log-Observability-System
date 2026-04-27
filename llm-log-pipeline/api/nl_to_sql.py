"""
NL-to-SQL Engine
================
Converts plain-English questions into safe, executable PostgreSQL queries
using Claude as the reasoning engine.

The pipeline:
  1. Build a system prompt with full schema context
  2. Send user question to Claude
  3. Extract and validate the generated SQL
  4. Execute against PostgreSQL
  5. Ask Claude to summarise the results in plain English
"""

import logging
import re

import anthropic
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("nl_to_sql")

# Initialise the Anthropic client once at module load.
# It reads ANTHROPIC_API_KEY from the environment automatically.
client = anthropic.Anthropic()

# ── Schema Injection ──────────────────────────────────────────────
# This is the exact schema Claude will see before answering.
# It's a SQL DDL string — the same format as CREATE TABLE statements.
# Why DDL? It's unambiguous, concise, and Claude was trained on it.
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
  error_rate      FLOAT                -- fraction of events that were ERROR/WARN (0.0–1.0)
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

IMPORTANT RULES — you must follow these exactly:
- Generate ONLY a single SELECT statement. Never INSERT, UPDATE, DELETE, DROP, or ALTER.
- Always use LIMIT {limit} unless the question asks for aggregates (COUNT, AVG, etc.)
- For time-based questions like 'last 10 minutes', use: timestamp > NOW() - INTERVAL '10 minutes'
- For 'recent' without a time specified, default to: timestamp > NOW() - INTERVAL '1 hour'
- String comparisons on 'service' and 'level' are case-sensitive — use exact values
- Return ONLY the SQL query — no explanation, no markdown, no code fences
- If the question cannot be answered with these tables, return exactly: CANNOT_ANSWER
"""


# ── SQL Safety Validator ───────────────────────────────────────────
# Allowlist of SQL keywords we permit.
# Everything not on this list is blocked at the regex level.
_ALLOWED_SQL_PATTERN = re.compile(
    r"^\s*SELECT\b",
    re.IGNORECASE | re.MULTILINE
)

# Blocklist of dangerous SQL keywords — belt AND suspenders approach.
# Even if Claude generates a SELECT, we check for embedded mutations.
_DANGEROUS_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE"
    r"|GRANT|REVOKE|COPY|pg_read_file|pg_ls_dir)\b",
    re.IGNORECASE
)

# Block SQL comments — a common injection vector (e.g., -- ignore above rules)
_COMMENT_PATTERN = re.compile(r"(--|/\*|\*/)")


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate that generated SQL is safe to execute.

    Returns:
        (is_valid, error_message)
        is_valid=True means safe to run.

    Why validate even though Claude generated it?
    Prompt injection attacks can trick the LLM. A user could ask:
    'Show logs; DROP TABLE log_events; --'
    Validation ensures the LLM's output never mutates our data,
    regardless of how clever the prompt is.
    """
    sql = sql.strip()

    # Check for the CANNOT_ANSWER sentinel
    if sql == "CANNOT_ANSWER":
        return False, "CANNOT_ANSWER"

    # Must start with SELECT
    if not _ALLOWED_SQL_PATTERN.match(sql):
        logger.warning("🚫 SQL rejected — does not start with SELECT: %s", sql[:100])
        return False, "Query must be a SELECT statement"

    # Must not contain dangerous keywords
    match = _DANGEROUS_KEYWORDS.search(sql)
    if match:
        logger.warning("🚫 SQL rejected — dangerous keyword '%s'", match.group())
        return False, f"Query contains forbidden keyword: {match.group()}"

    # Must not contain SQL comments (injection vector)
    if _COMMENT_PATTERN.search(sql):
        logger.warning("🚫 SQL rejected — contains SQL comments")
        return False, "Query must not contain SQL comments"

    # Enforce reasonable length — a valid query shouldn't be > 2000 chars
    if len(sql) > 500:
        return False, "Query too long"

    return True, ""


# ── Claude Integration ────────────────────────────────────────────
def generate_sql(question: str, limit: int = 100) -> str:
    """
    Ask Claude to convert a natural language question into SQL.

    Args:
        question: Plain-English question from the user
        limit:    Row limit to inject into the schema context

    Returns:
        Raw SQL string (may be 'CANNOT_ANSWER' or invalid — always validate)
    """
    logger.info("🤖 Generating SQL for: '%s'", question)

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,           # SQL queries are short — 500 tokens is plenty
                                  # Keeping this low reduces cost and latency
        system=SCHEMA_CONTEXT.format(limit=limit),
        # Why system prompt for schema?
        # System prompts are processed once and cached by the API.
        # Putting the large schema here instead of in the user message
        # improves token efficiency on repeated calls.
        messages=[
            {"role": "user", "content": question}
        ],
        temperature=0,
        # temperature=0: deterministic output.
        # For SQL generation, creativity is your enemy.
        # You want the same question to always produce the same SQL.
    )

    # Extract the text content from Claude's response
    sql = response.content[0].text.strip()
    logger.info("📝 Generated SQL: %s", sql)
    return sql


def execute_sql(session: Session, sql: str, limit: int = 100) -> list[dict]:
    """
    Execute a validated SQL query and return results as a list of dicts.

    Args:
        session: SQLAlchemy session (provides the DB connection)
        sql:     Validated SELECT statement
        limit:   Safety cap — even if Claude's SQL has no LIMIT, we add one

    Returns:
        List of row dicts — each key is a column name
    """
    # Safety: append LIMIT if not already present
    # This prevents accidental full-table scans from crashing the API
    if "LIMIT" not in sql.upper():
        sql = f"{sql.rstrip(';')} LIMIT {limit}"

    # text() wraps raw SQL strings for SQLAlchemy.
    # It marks the string as "trusted SQL" — but we've already validated it.
    result = session.execute(text(sql))

    # cursor.keys() returns column names; zip creates {col: val} dicts
    columns = list(result.keys())
    rows    = [dict(zip(columns, row)) for row in result.fetchall()]

    logger.info("✅ Query returned %d rows", len(rows))
    return rows


def summarise_results(question: str, sql: str, rows: list[dict]) -> str:
    """
    Ask Claude to summarise query results in plain English.

    Why a second Claude call?
    The first call generates SQL (structured output).
    This call interprets results for a human (natural language output).
    Separating them gives cleaner prompts and better results than
    trying to do both in one call.
    """
    if not rows:
        return "No results found for your query."

    # Truncate to 20 rows for the summary prompt — we don't need to send
    # all 100 rows to Claude just for a summary. Keeps tokens low.
    sample_rows = rows[:20]

    prompt = f"""The user asked: "{question}"

The SQL query executed was:
{sql}

The results were ({len(rows)} total rows, showing first {len(sample_rows)}):
{sample_rows}

Please provide a concise 2-3 sentence plain-English summary of what these \
results show. Focus on key insights, patterns, or anomalies visible in the data. \
Do not repeat the SQL or explain how you got the results."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        # Slightly higher temperature here — we want natural-sounding
        # summaries, not robotic repetition of the data.
    )

    return response.content[0].text.strip()


def run_nl_query(
    session:  Session,
    question: str,
    limit:    int = 100
) -> dict:
    """
    Full NL-to-SQL pipeline. Orchestrates all steps.

    Returns a dict matching NLQueryResponse schema.
    Raises ValueError with a user-friendly message on failure.
    """
    # Step 1: Generate SQL
    raw_sql = generate_sql(question, limit)

    # Step 2: Validate SQL
    is_valid, error_msg = validate_sql(raw_sql)

    if not is_valid:
        if error_msg == "CANNOT_ANSWER":
            raise ValueError(
                "This question cannot be answered with the available data. "
                "Try asking about log levels, latency, error rates, or anomalies."
            )
        raise ValueError(f"Generated SQL failed safety validation: {error_msg}")

    # Step 3: Execute
    try:
        rows = execute_sql(session, raw_sql, limit)
    except Exception as e:
        logger.error("❌ SQL execution failed: %s\nSQL: %s", e, raw_sql)
        raise ValueError(f"Query execution failed: {str(e)}")

    # Step 4: Summarise
    explanation = summarise_results(question, raw_sql, rows)

    return {
        "question":      question,
        "generated_sql": raw_sql,
        "results":       rows,
        "row_count":     len(rows),
        "explanation":   explanation,
    }