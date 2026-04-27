# LLM-Augmented Log Observability Pipeline

Real-time anomaly detection and natural language querying over streaming logs — a self-hosted alternative to Datadog with GenAI querying powered by Claude.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Kafka](https://img.shields.io/badge/Kafka-7.5-black)
![Claude](https://img.shields.io/badge/Claude-API-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

## What This Project Does

Engineers spend hours manually sifting through logs during incidents. Commercial tools like Datadog cost thousands per month. This project builds a self-hosted pipeline that:

- **Ingests** streaming logs from 3 simulated microservices via Apache Kafka
- **Detects anomalies** automatically using Isolation Forest ML on 30-second windows
- **Answers questions** in plain English via a Claude-powered NL-to-SQL interface
- **Visualises** everything on a live Streamlit dashboard

## Architecture

```
┌─────────────┐     ┌─────────────┐    ┌──────────────────┐
│ Log Producer│───▶│Apache Kafka │───▶│ Stream Consumer  │
│ (3 services)│     │  app-logs   │    │ Isolation Forest │
└─────────────┘     └─────────────┘    └────────┬─────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │  PostgreSQL  │
                                        │  log_events  │
                                        │anomaly_alerts│
                                        └──────┬───────┘
                                               │
                          ┌────────────────────┤
                          ▼                    ▼
                   ┌─────────────┐     ┌──────────────────┐
                   │   FastAPI   │◀──▶│   Claude API     │
                   │   Backend   │     │    NL-to-SQL     │
                   └──────┬──────┘     └──────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Streamlit  │
                   │  Dashboard  │
                   └─────────────┘
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Log Generation | Python, Faker | Simulates realistic microservice traffic |
| Message Bus | Apache Kafka | Durable, replayable event stream |
| Anomaly Detection | scikit-learn Isolation Forest | Unsupervised ML |
| Storage | PostgreSQL + SQLAlchemy | Persists logs and anomaly scores |
| API | FastAPI + Pydantic v2 | REST endpoints + request validation |
| LLM Integration | Anthropic Claude API | NL-to-SQL with schema injection |
| Dashboard | Streamlit + Plotly | Live charts and query interface |
| Infrastructure | Docker Compose | Single-command deployment |

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 enabled
- [Anthropic API key](https://console.anthropic.com) (free credits on signup)

### 1 — Clone and configure

```bash
git clone https://github.com/yourusername/llm-log-pipeline
cd llm-log-pipeline
cp .env.example .env
```

Open `.env` and set your API key:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

### 2 — Launch

```bash
docker compose up -d
```

Wait approximately 90 seconds for Kafka to fully initialise.

### 3 — Access

| Interface | URL |
|---|---|
| 📊 Live Dashboard | http://localhost:8501 |
| 📖 API Docs | http://localhost:8000/docs |
| ❤️ API Health | http://localhost:8000/health |

### 4 — Try a natural language query

On the dashboard, type into the query box:
> *"Which service has the highest error rate in the last 10 minutes?"*

Claude generates the SQL, executes it safely, and returns a plain-English summary.

## Running Tests

```bash
cd tests
pip install -r requirements-dev.txt
pytest .. -v --tb=short --ignore=../dashboard
```

Expected: **56 tests passing**

## Key Design Decisions

**Kafka over a simple queue** — Kafka's durable log means the consumer can replay messages after a crash. No logs are lost even if the consumer restarts mid-incident.

**Isolation Forest for anomaly detection** — Fully unsupervised. No labelled anomaly data needed. Handles high-dimensional log feature vectors naturally, and retrains automatically as traffic patterns evolve.

**Schema injection for NL-to-SQL** — Claude receives the exact PostgreSQL DDL before answering. This prevents hallucinated column names. All generated SQL is validated against a strict allowlist before execution — no mutations possible.

**Per-service anomaly windows** — Each microservice maintains its own independent 30-second tumbling window. A payment spike doesn't pollute the auth service baseline.

## Project Structure

```
llm-log-pipeline/
├── producer/          # Kafka log producer — simulates 3 microservices
│   ├── main.py        # Log generation + fault injection scheduler
│   ├── Dockerfile
│   └── requirements.txt
├── consumer/          # Kafka consumer + ML anomaly scorer
│   ├── main.py        # Window manager + pipeline orchestration
│   ├── features.py    # Feature extraction from log windows
│   ├── scorer.py      # Isolation Forest wrapper with auto-retraining
│   ├── models.py      # SQLAlchemy ORM models
│   ├── Dockerfile
│   └── requirements.txt
├── api/               # FastAPI backend
│   ├── main.py        # REST endpoints
│   ├── nl_to_sql.py   # Claude NL-to-SQL engine + SQL validator
│   ├── schemas.py     # Pydantic request/response schemas
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/         # Streamlit live dashboard
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── tests/             # pytest suite (56 tests)
├── docs/              # Architecture diagram
├── docker-compose.yml
├── .env.example
└── README.md
```

