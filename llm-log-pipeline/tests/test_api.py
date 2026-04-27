"""
Tests for FastAPI endpoints.

Root causes fixed in this version:
1. Removed patch of declarative_base — let SQLAlchemy ORM classes load
   normally so LogEvent.timestamp.desc() works without AttributeError.
2. Used Annotated + StringConstraints in schemas.py for proper Pydantic v2
   whitespace stripping — fixing the whitespace-only 500 error.
3. Removed all patch.object(api.LogEvent, 'timestamp') calls — no longer
   needed since LogEvent is now a real SQLAlchemy class.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Environment setup BEFORE any module loads ─────────────────────
os.environ["POSTGRES_USER"]     = "testuser"
os.environ["POSTGRES_PASSWORD"] = "testpass"
os.environ["POSTGRES_HOST"]     = "localhost"
os.environ["POSTGRES_PORT"]     = "5432"
os.environ["POSTGRES_DB"]       = "testdb"
os.environ["ANTHROPIC_API_KEY"] = "test-key-placeholder"

API_MAIN_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'api', 'main.py')
)


def load_api_module():
    """
    Load api/main.py by explicit file path to avoid name collision
    with producer/main.py.

    Key change from previous version:
    We NO LONGER patch declarative_base. Patching it turned LogEvent
    into a MagicMock(spec='str'), which broke LogEvent.timestamp.desc().

    We only patch create_engine — enough to prevent real DB connections
    while keeping the SQLAlchemy ORM class definitions intact.
    """
    for key in list(sys.modules.keys()):
        if key in ('main', 'api_main'):
            del sys.modules[key]

    spec   = importlib.util.spec_from_file_location("api_main", API_MAIN_PATH)
    module = importlib.util.module_from_spec(spec)

    # Patch ONLY create_engine — stops DB connection attempts.
    # declarative_base is left alone so LogEvent, AnomalyAlert remain
    # real SQLAlchemy classes with working column descriptors.
    with patch("sqlalchemy.create_engine") as mock_engine:
        mock_engine.return_value = MagicMock()
        spec.loader.exec_module(module)

    return module


def make_mock_session():
    """
    Build a fully-chained SQLAlchemy session mock.

    Chain: db.query().order_by().filter().limit().all()
    Each link returns mock_chain so the chain never breaks.
    """
    mock_session = MagicMock()
    mock_chain   = MagicMock()

    mock_session.query.return_value  = mock_chain
    mock_chain.order_by.return_value = mock_chain
    mock_chain.filter.return_value   = mock_chain
    mock_chain.limit.return_value    = mock_chain
    mock_chain.all.return_value      = []

    return mock_session, mock_chain


# ── Health Endpoint ───────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_ok(self):
        """
        /health must return 200 with {status: ok}.
        No DB dependency — always succeeds.
        """
        from fastapi.testclient import TestClient
        api    = load_api_module()
        client = TestClient(api.app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


# ── Logs Endpoint ─────────────────────────────────────────────────

class TestLogsEndpoint:

    def test_logs_returns_200(self):
        """
        GET /logs must respond HTTP 200.

        With declarative_base no longer patched, LogEvent is a real
        SQLAlchemy class. LogEvent.timestamp.desc() works correctly
        as a real column descriptor — no patching needed.
        """
        from fastapi.testclient import TestClient
        api             = load_api_module()
        mock_session, _ = make_mock_session()
        api.app.dependency_overrides[api.get_db] = lambda: mock_session

        client   = TestClient(api.app)
        response = client.get("/logs")

        assert response.status_code == 200

    def test_logs_returns_list(self):
        """GET /logs response body must be a JSON array."""
        from fastapi.testclient import TestClient
        api             = load_api_module()
        mock_session, _ = make_mock_session()
        api.app.dependency_overrides[api.get_db] = lambda: mock_session

        client   = TestClient(api.app)
        response = client.get("/logs")

        assert isinstance(response.json(), list)

    def test_logs_limit_param_validation(self):
        """
        limit=0 → 422 (violates ge=1)
        limit=1001 → 422 (violates le=1000)
        FastAPI validates these before the endpoint function runs.
        """
        from fastapi.testclient import TestClient
        api             = load_api_module()
        mock_session, _ = make_mock_session()
        api.app.dependency_overrides[api.get_db] = lambda: mock_session

        client = TestClient(api.app)

        response = client.get("/logs?limit=0")
        assert response.status_code == 422, (
            f"limit=0 should be rejected, got {response.status_code}"
        )

        response = client.get("/logs?limit=1001")
        assert response.status_code == 422, (
            f"limit=1001 should be rejected, got {response.status_code}"
        )

    def test_logs_service_filter_applied(self):
        """
        ?service=auth must cause .filter() to be called on the query.
        Confirms the filter logic isn't silently skipped.
        """
        from fastapi.testclient import TestClient
        api                      = load_api_module()
        mock_session, mock_chain = make_mock_session()
        api.app.dependency_overrides[api.get_db] = lambda: mock_session

        client = TestClient(api.app)
        client.get("/logs?service=auth")

        mock_chain.filter.assert_called()


# ── Query Endpoint ────────────────────────────────────────────────

class TestQueryEndpoint:

    def test_query_rejects_empty_question(self):
        """
        Empty string and whitespace-only must both return 422.

        Requires schemas.py to use:
            Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

        This is the Pydantic v2 correct way — Field(strip_whitespace=True)
        was deprecated and silently ignored, letting whitespace through.
        """
        from fastapi.testclient import TestClient
        api          = load_api_module()
        mock_session = MagicMock()
        api.app.dependency_overrides[api.get_db] = lambda: mock_session

        client = TestClient(api.app)

        response = client.post("/query", json={"question": "", "limit": 10})
        assert response.status_code == 422, (
            f"Empty string should return 422, got {response.status_code}"
        )

        response = client.post("/query", json={"question": "   ", "limit": 10})
        assert response.status_code == 422, (
            f"Whitespace-only should return 422, got {response.status_code}"
        )

    def test_query_returns_expected_fields(self):
        """
        Successful NL query must return all 5 NLQueryResponse fields.
        run_nl_query is mocked — no real Claude API call made.
        """
        from fastapi.testclient import TestClient
        api          = load_api_module()
        mock_session = MagicMock()
        api.app.dependency_overrides[api.get_db] = lambda: mock_session

        mock_result = {
            "question":      "How many logs are there?",
            "generated_sql": "SELECT COUNT(*) FROM log_events",
            "results":       [{"count": 42}],
            "row_count":     1,
            "explanation":   "There are 42 log events in the database.",
        }

        with patch.object(api, 'run_nl_query', return_value=mock_result):
            client   = TestClient(api.app)
            response = client.post(
                "/query",
                json={"question": "How many logs are there?", "limit": 10}
            )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        for field in ("question", "generated_sql", "results", "row_count", "explanation"):
            assert field in data, f"Missing field: '{field}'"

    def test_query_handles_claude_api_error_gracefully(self):
        """
        Unexpected errors from Claude must return 500, not crash the server.
        """
        from fastapi.testclient import TestClient
        api          = load_api_module()
        mock_session = MagicMock()
        api.app.dependency_overrides[api.get_db] = lambda: mock_session

        with patch.object(
            api, 'run_nl_query',
            side_effect=RuntimeError("Claude API unavailable")
        ):
            client   = TestClient(api.app)
            response = client.post(
                "/query",
                json={"question": "Show me recent errors", "limit": 10}
            )

        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]