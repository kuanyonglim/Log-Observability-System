"""
Streamlit Dashboard
===================
Live-updating observability dashboard.
Talks exclusively to the FastAPI backend — never directly to the DB.

Layout:
  ┌─────────────────────────────────────┐
  │  Header: pipeline health metrics    │
  ├──────────────┬──────────────────────┤
  │  Log volume  │  Anomaly score       │
  │  by service  │  time series         │
  ├──────────────┴──────────────────────┤
  │  Live anomaly alert feed            │
  ├─────────────────────────────────────┤
  │  Natural language query box         │
  └─────────────────────────────────────┘
"""

import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

# ── Config ────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000")
# Inside Docker, services talk to each other by container name.
# 'api' resolves to the FastAPI container's IP automatically.
# When running locally (outside Docker), change to http://localhost:8000

REFRESH_INTERVAL = 10  # seconds between auto-refreshes

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Log Observability Pipeline",
    page_icon="🔍",
    layout="wide",          # use full browser width
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────
# Small style tweaks — Streamlit's defaults are functional but plain
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #7c3aed;
    }
    .anomaly-alert {
        background-color: #2d1b1b;
        border-left: 4px solid #ef4444;
        padding: 10px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .normal-window {
        background-color: #1b2d1b;
        border-left: 4px solid #22c55e;
        padding: 10px;
        border-radius: 4px;
        margin: 4px 0;
    }
    .stMetric label { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)


# ── API Helper ────────────────────────────────────────────────────
def fetch(endpoint: str, params: dict = None) -> dict | list | None:
    """
    Make a GET request to the FastAPI backend.

    Returns None on failure instead of raising — the dashboard
    should degrade gracefully if the API is temporarily unavailable,
    not crash with a stack trace.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            params=params,
            timeout=5,  # don't hang the dashboard if API is slow
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.warning(f"⚠️ Cannot reach API at {API_BASE_URL}. Is it running?")
        return None
    except requests.exceptions.Timeout:
        st.warning("⚠️ API request timed out.")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def post(endpoint: str, body: dict) -> dict | None:
    """Make a POST request to the FastAPI backend."""
    try:
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=body,
            timeout=30,  # NL queries can take longer (Claude API call)
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot reach API at {API_BASE_URL}")
        return None
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None


# ── Header ────────────────────────────────────────────────────────
def render_header(metrics: dict | None) -> None:
    """Top bar with pipeline health KPIs."""
    st.title("🔍 LLM Log Observability Pipeline")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')} · Auto-refreshes every {REFRESH_INTERVAL}s")

    if not metrics:
        st.warning("Waiting for pipeline data...")
        return

    # Four KPI columns across the top
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📨 Total Logs Ingested",
            value=f"{metrics.get('total_logs_ingested', 0):,}",
        )
    with col2:
        st.metric(
            label="🪟 Windows Scored",
            value=f"{metrics.get('total_windows_scored', 0):,}",
        )
    with col3:
        st.metric(
            label="🚨 Anomalies Detected",
            value=f"{metrics.get('total_anomalies_flagged', 0):,}",
        )
    with col4:
        anomaly_rate = metrics.get('anomaly_rate_pct', 0)
        # Colour the metric red if anomaly rate is high
        st.metric(
            label="📊 Anomaly Rate",
            value=f"{anomaly_rate:.1f}%",
            delta=f"{'⚠️ High' if anomaly_rate > 10 else '✅ Normal'}",
            delta_color="inverse",
        )

    # Last anomaly banner
    last = metrics.get("last_anomaly")
    if last:
        st.error(
            f"🔴 Last anomaly: **{last['service']}** service at "
            f"{last['window_start'][:19].replace('T', ' ')} UTC "
            f"(score: {last['score']:.4f})"
        )


# ── Log Volume Chart ──────────────────────────────────────────────
def render_log_volume(logs: list | None) -> None:
    """Bar chart: log counts per service, coloured by level."""
    st.subheader("📊 Log Volume by Service")

    if not logs:
        st.info("No log data yet. Waiting for pipeline...")
        return

    df = pd.DataFrame(logs)

    if df.empty:
        st.info("No logs in the selected window.")
        return

    # Count logs per (service, level) combination
    counts = (
        df.groupby(["service", "level"])
        .size()
        .reset_index(name="count")
    )

    # Consistent colour mapping for log levels
    color_map = {
        "INFO":  "#22c55e",   # green
        "WARN":  "#f59e0b",   # amber
        "ERROR": "#ef4444",   # red
    }

    fig = px.bar(
        counts,
        x="service",
        y="count",
        color="level",
        color_discrete_map=color_map,
        barmode="stack",        # stacked so total volume is visible
        title="Log events by service and level",
        labels={"count": "Events", "service": "Service", "level": "Level"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend_title="Log Level",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Anomaly Score Time Series ─────────────────────────────────────
def render_anomaly_timeseries(anomalies: list | None) -> None:
    """
    Line chart: anomaly scores over time per service.
    Anomalous windows are highlighted with red markers.
    """
    st.subheader("📈 Anomaly Scores Over Time")

    if not anomalies:
        st.info("No anomaly windows yet. Waiting for first 30s window...")
        return

    df = pd.DataFrame(anomalies)

    if df.empty:
        st.info("No anomaly data available.")
        return

    df["window_start"] = pd.to_datetime(df["window_start"])

    # One line per service
    fig = go.Figure()
    colors = {
        "auth":        "#7c3aed",
        "payment":     "#2563eb",
        "api-gateway": "#0891b2",
    }

    for service in df["service"].unique():
        svc_df = df[df["service"] == service].sort_values("window_start")

        # Normal windows — thin line
        fig.add_trace(go.Scatter(
            x=svc_df["window_start"],
            y=svc_df["anomaly_score"],
            mode="lines",
            name=service,
            line=dict(color=colors.get(service, "#94a3b8"), width=2),
        ))

        # Anomalous windows — large red markers on top
        anomalous = svc_df[svc_df["is_anomaly"] == True]
        if not anomalous.empty:
            fig.add_trace(go.Scatter(
                x=anomalous["window_start"],
                y=anomalous["anomaly_score"],
                mode="markers",
                name=f"{service} (anomaly)",
                marker=dict(color="#ef4444", size=12, symbol="x"),
                showlegend=True,
            ))

    # Threshold line — visually shows the anomaly decision boundary
    fig.add_hline(
        y=-0.1,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text="Anomaly threshold",
        annotation_position="bottom right",
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        xaxis_title="Time",
        yaxis_title="Anomaly Score (lower = more anomalous)",
        height=350,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Alert Feed ────────────────────────────────────────────────────
def render_alert_feed(anomalies: list | None) -> None:
    """Scrollable feed showing the most recent anomaly windows."""
    st.subheader("🚨 Anomaly Alert Feed")

    if not anomalies:
        st.info("No anomaly data yet.")
        return

    # Show only the 15 most recent windows
    recent = anomalies[:15]

    for alert in recent:
        window_time = alert["window_start"][:19].replace("T", " ")
        is_anomaly  = alert.get("is_anomaly", False)

        if is_anomaly:
            # Red card for anomalies
            st.markdown(f"""
<div class="anomaly-alert">
🔴 <strong>{alert['service'].upper()}</strong> · {window_time} UTC<br>
Score: <strong>{alert.get('anomaly_score', 0):.4f}</strong> &nbsp;|&nbsp;
Error rate: <strong>{alert.get('error_rate', 0)*100:.1f}%</strong> &nbsp;|&nbsp;
Avg latency: <strong>{alert.get('avg_latency_ms', 0):.0f}ms</strong> &nbsp;|&nbsp;
Events: <strong>{alert.get('event_count', 0)}</strong>
</div>
""", unsafe_allow_html=True)
        else:
            # Green card for normal windows
            st.markdown(f"""
<div class="normal-window">
✅ <strong>{alert['service'].upper()}</strong> · {window_time} UTC<br>
Score: {alert.get('anomaly_score', 0):.4f} &nbsp;|&nbsp;
Error rate: {alert.get('error_rate', 0)*100:.1f}% &nbsp;|&nbsp;
Avg latency: {alert.get('avg_latency_ms', 0):.0f}ms
</div>
""", unsafe_allow_html=True)


# ── NL Query Box ──────────────────────────────────────────────────
def render_nl_query() -> None:
    """
    Natural language query interface.
    Sends the question to POST /query and displays the results.
    """
    st.subheader("💬 Ask a Question in Plain English")

    # Example questions as clickable buttons
    st.caption("Try one of these:")
    example_cols = st.columns(3)
    examples = [
        "Show me the last 10 ERROR logs from payment",
        "Which service has the highest error rate in the last hour?",
        "How many anomalies were detected today?",
    ]

    # Session state tracks which example was clicked
    # (Streamlit reruns on every interaction — state persists across reruns)
    if "nl_question" not in st.session_state:
        st.session_state.nl_question = ""

    for i, (col, example) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.nl_question = example

    # Main query input
    question = st.text_input(
        label="Your question:",
        value=st.session_state.nl_question,
        placeholder="e.g. What's the p99 latency for auth service right now?",
        key="nl_input",
    )

    col_btn, col_limit = st.columns([3, 1])
    with col_limit:
        limit = st.number_input("Row limit", min_value=1, max_value=500, value=50)
    with col_btn:
        run_query = st.button("🔍 Run Query", type="primary", use_container_width=True)

    if run_query and question.strip():
        with st.spinner("🤖 Claude is generating SQL..."):
            result = post("/query", {"question": question, "limit": limit})

        if result:
            # Show the generated SQL — transparency is key for trust
            with st.expander("📝 Generated SQL (click to expand)", expanded=True):
                st.code(result["generated_sql"], language="sql")

            # Claude's plain-English summary
            st.info(f"💡 **Summary:** {result['explanation']}")

            # Results table
            st.caption(f"Returned {result['row_count']} rows")
            if result["results"]:
                df = pd.DataFrame(result["results"])
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("Query returned no results.")


# ── Latency Distribution ──────────────────────────────────────────
def render_latency_distribution(logs: list | None) -> None:
    """Box plot showing latency spread per service."""
    st.subheader("⏱️ Latency Distribution by Service")

    if not logs:
        return

    df = pd.DataFrame(logs)
    if df.empty or "latency_ms" not in df.columns:
        return

    fig = px.box(
        df,
        x="service",
        y="latency_ms",
        color="service",
        color_discrete_map={
            "auth":        "#7c3aed",
            "payment":     "#2563eb",
            "api-gateway": "#0891b2",
        },
        title="Request latency distribution (ms)",
        labels={"latency_ms": "Latency (ms)", "service": "Service"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        showlegend=False,
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Main App ──────────────────────────────────────────────────────
def main() -> None:
    # Fetch all data upfront — one round of API calls per refresh
    metrics   = fetch("/metrics")
    logs      = fetch("/logs",      params={"limit": 500})
    anomalies = fetch("/anomalies", params={"limit": 100})

    # ── Layout ────────────────────────────────────────────────────
    render_header(metrics)
    st.divider()

    # Row 1: two charts side by side
    left_col, right_col = st.columns([1, 1])
    with left_col:
        render_log_volume(logs)
    with right_col:
        render_latency_distribution(logs)

    st.divider()

    # Row 2: full-width anomaly time series
    render_anomaly_timeseries(anomalies)

    st.divider()

    # Row 3: alert feed + NL query side by side
    feed_col, query_col = st.columns([1, 1])
    with feed_col:
        render_alert_feed(anomalies)
    with query_col:
        render_nl_query()

    # ── Auto-refresh ──────────────────────────────────────────────
    # st.rerun() triggers a full script rerun after REFRESH_INTERVAL seconds.
    # This is Streamlit's way of doing live updates — no WebSockets needed.
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()