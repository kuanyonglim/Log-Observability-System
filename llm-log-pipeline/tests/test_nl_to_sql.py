"""
Tests for the NL-to-SQL safety validation layer.

We test the validator exhaustively because it's a security boundary.
Every bypass here means a user could potentially corrupt the database.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'api'))

from nl_to_sql import validate_sql


class TestValidateSQL:
    """Tests for the SQL safety validator."""

    # ── Valid queries that SHOULD pass ───────────────────────────

    def test_accepts_simple_select(self):
        sql = "SELECT * FROM log_events LIMIT 10"
        is_valid, err = validate_sql(sql)
        assert is_valid, f"Simple SELECT should be valid, got: {err}"

    def test_accepts_select_with_where(self):
        sql = "SELECT * FROM log_events WHERE service = 'auth' LIMIT 20"
        is_valid, err = validate_sql(sql)
        assert is_valid, f"SELECT with WHERE should be valid, got: {err}"

    def test_accepts_aggregate_query(self):
        sql = """
            SELECT service, COUNT(*) as count, AVG(latency_ms) as avg_latency
            FROM log_events
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            GROUP BY service
        """
        is_valid, err = validate_sql(sql)
        assert is_valid, f"Aggregate SELECT should be valid, got: {err}"

    def test_accepts_join_query(self):
        sql = """
            SELECT l.service, l.level, a.anomaly_score
            FROM log_events l
            JOIN anomaly_alerts a ON l.service = a.service
            LIMIT 10
        """
        is_valid, err = validate_sql(sql)
        assert is_valid, f"JOIN query should be valid, got: {err}"

    def test_accepts_cannot_answer_sentinel(self):
        """CANNOT_ANSWER is a valid sentinel — validator handles it gracefully."""
        is_valid, err = validate_sql("CANNOT_ANSWER")
        assert not is_valid
        assert err == "CANNOT_ANSWER"

    # ── Dangerous queries that MUST be rejected ───────────────────

    def test_rejects_drop_table(self):
        is_valid, err = validate_sql("DROP TABLE log_events")
        assert not is_valid, "DROP TABLE must be rejected"

    def test_rejects_delete(self):
        is_valid, err = validate_sql("DELETE FROM log_events WHERE 1=1")
        assert not is_valid, "DELETE must be rejected"

    def test_rejects_insert(self):
        is_valid, err = validate_sql(
            "INSERT INTO log_events (service) VALUES ('hacked')"
        )
        assert not is_valid, "INSERT must be rejected"

    def test_rejects_update(self):
        is_valid, err = validate_sql(
            "UPDATE log_events SET level = 'INFO' WHERE 1=1"
        )
        assert not is_valid, "UPDATE must be rejected"

    def test_rejects_truncate(self):
        is_valid, err = validate_sql("TRUNCATE TABLE log_events")
        assert not is_valid, "TRUNCATE must be rejected"

    def test_rejects_sql_comment_injection(self):
        """
        A classic injection: valid-looking start, malicious payload after --.
        The -- comments out everything after it in SQL.
        """
        is_valid, err = validate_sql(
            "SELECT * FROM log_events -- DROP TABLE log_events"
        )
        assert not is_valid, "SQL comment injection must be rejected"

    def test_rejects_semicolon_chaining(self):
        """
        Semicolons allow chaining multiple statements.
        Even if the first is SELECT, the second could be DROP.
        Our regex catches this because DROP appears in the string.
        """
        is_valid, err = validate_sql(
            "SELECT * FROM log_events; DROP TABLE log_events"
        )
        assert not is_valid, "Semicolon-chained DROP must be rejected"

    def test_rejects_embedded_delete_in_select(self):
        """DELETE embedded inside what looks like a SELECT context."""
        is_valid, err = validate_sql(
            "SELECT * FROM log_events WHERE id IN (DELETE FROM log_events)"
        )
        assert not is_valid, "Embedded DELETE must be rejected"

    def test_rejects_pg_system_functions(self):
        """PostgreSQL system functions that could read server files."""
        is_valid, err = validate_sql(
            "SELECT pg_read_file('/etc/passwd')"
        )
        assert not is_valid, "pg_read_file must be rejected"

    def test_rejects_empty_string(self):
        """Empty input is not a valid SELECT."""
        is_valid, err = validate_sql("")
        assert not is_valid, "Empty string must be rejected"

    def test_rejects_case_insensitive_drop(self):
        """
        Attackers often use mixed case to bypass naive string matching.
        Our regex uses re.IGNORECASE so dRoP tAbLe is still caught.
        """
        is_valid, err = validate_sql("select * from log_events; dRoP tAbLe users")
        assert not is_valid, "Case-insensitive DROP must be rejected"


class TestSQLLengthLimit:

    def test_rejects_extremely_long_query(self):
        """Queries over 2000 chars are suspicious — likely injection attempts."""
        long_sql = "SELECT " + "*, " * 500 + "1 FROM log_events"
        is_valid, err = validate_sql(long_sql)
        assert not is_valid, "Excessively long query must be rejected"