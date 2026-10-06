# AIO v3 — Agent Runtime & Reliability Layer

[![Tests](https://img.shields.io/badge/tests-61%2F61%20passed-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.x-blue)]() [![Runtime](https://img.shields.io/badge/runtime-agentic-orange)]()

> A production-oriented experimental runtime for building more reliable, observable and controllable LLM agents.

AIO v3 is an agent runtime designed to address practical engineering problems that appear when LLM applications move from simple inference to multi-step, tool-using and stateful agents.

The runtime combines model routing, semantic caching, guardrails, structured-output validation, persistent memory, tool execution, multi-tenancy, observability, resilience and state rollback.

---

## Why AIO v3?

A basic LLM application often looks like:

```text
User → LLM → Response

An agentic system is more complex:
User
 ↓
Routing
 ↓
LLM
 ↓
Tools / Memory / External services
 ↓
Validation
 ↓
Recovery / Retry
 ↓
Observable state

AIO v3 provides a runtime layer around the model to control, validate and observe this execution flow.
The goal is not to replace the underlying LLM. The goal is to make the surrounding system more reliable, observable and controllable.
Core capabilities
| # | Capability | Purpose |
|---|---|---|
| 1 | Semantic cosine cache | Avoid unnecessary repeated inference |
| 2 | Native token streaming | Low-latency SSE responses |
| 3 | Multi-provider circuit breaker | Prevent repeated provider failures |
| 4 | Complexity-based model selector | Route requests according to complexity |
| 5 | Prompt Shield | Detect / mitigate unsafe prompt patterns |
| 6 | JSON validation + retry | Enforce structured outputs |
| 7 | Persistent SQLite memory | Maintain agent state |
| 8 | Tool use | Web / Python / file execution |
| 9 | Multi-tenancy + quotas | Isolate clients and control usage |
| 10 | Real-time dashboard | Runtime visibility |
| 11 | OpenTelemetry / Prometheus | Metrics and tracing |
| 12 | Automatic mutation rollback | Recover from failed state changes |
Architecture
                    ┌──────────────────┐
                    │      Client      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    AIO API       │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Semantic Cache   Model Router   Prompt Shield
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌──────────────────┐
                    │   LLM Provider   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
           Memory          Tools        Validation
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌──────────────────┐
                    │ Recovery / Retry │
                    │ Circuit Breaker  │
                    │    Rollback      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Observability    │
                    │ OTel / Prometheus│
                    └──────────────────┘

Reliability layer
AIO v3 separates model inference from runtime reliability.
The runtime can:
 * Validate structured responses
 * Retry invalid outputs
 * Detect provider failures
 * Switch between providers
 * Enforce quotas
 * Preserve persistent state
 * Expose telemetry
 * Rollback mutations when required
This architecture is intended for applications where an LLM response is not sufficient by itself and the surrounding execution state matters.
Performance validation
Initial Linux Cloud Shell measurements include:
| Component | Measured time |
|---|---|
| Prompt Shield / PII scrubbing | ~1 µs |
| Injection detection | ~1 µs |
| Model selection | ~0.7 µs |
| Circuit breaker operation | ~63 ns |
| JSON structural validation | ~144 µs |
| Multi-tenant provisioning | ~209 µs |
These measurements are implementation-level microbenchmarks and should not be interpreted as end-to-end LLM latency.
Reproduce benchmarks with:
python -m benchmarks.run_benchmarks

See the benchmarks/ directory for the measurement code.
Test status
Current validation:
 * 61 / 61 tests passed
 * 100% test pass rate
 * ~4.34 s total test execution
> Benchmark and test results are environment-dependent. Re-run the suite locally before using these numbers as a production performance claim.
> 
Quick start
git clone [https://github.com/albanfredon23/aiov3.git](https://github.com/albanfredon23/aiov3.git)
cd aiov3

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env

pytest tests/ -v

python -m aio.api.main

API:
http://localhost:8000

API
Query
POST /v1/query

Pipeline: cache → shield → model selector → LLM → validation
Streaming
POST /v1/stream

Native SSE token streaming.
Dashboard
GET /dashboard

Metrics
GET /metrics

Prometheus-compatible metrics.
Tenant management
POST /admin/tenants

Agent memory
GET /agents/{id}/memory

Mutation state
GET /mutations

Telemetry
GET /events/telemetry
WS  /ws/telemetry

Use cases
AIO v3 can serve as an experimental runtime layer for:
 * LLM agents
 * Multi-model applications
 * Tool-using agents
 * AI APIs
 * Autonomous workflows
 * Internal AI platforms
 * Applications requiring structured outputs
 * Applications requiring runtime observability
Design principles
 * Model-agnostic runtime
   AIO is designed around the runtime surrounding the model rather than being tied to a single LLM provider.
 * Controlled execution
   Agent actions should be observable, validated and recoverable.
 * Explicit state
   Memory and mutations are treated as runtime state rather than implicit side effects.
 * Measurable behavior
   Performance and reliability should be evaluated through reproducible tests and benchmarks.
Project status
AIO v3 is an experimental engineering project focused on agent runtime architecture, reliability and observability.
The project is actively evolving. Performance figures should be considered preliminary until independently reproduced across multiple environments.
Commercial / integration
AIO v3 can be integrated or extended for custom agent-runtime projects, including:
 * Agent runtime integration
 * Model routing
 * Reliability and recovery layers
 * AI observability
 * Tool-use infrastructure
 * Custom AI runtime development
For technical collaboration or integration requests, contact the repository author through GitHub.
Author
Alban Fredon
AI / Python / Agent Runtime Engineering
GitHub: https://github.com/albanfredon23


---

## AIOTrade (gestion des risques et trading algorithmique)

Le dossier [`aiotrade/`](aiotrade/README.md) contient un projet autonome : moteur de scénarios TAP, garde-fou
financier SCG, filtre χ² de rupture de marché, API FastAPI et site vitrine 3D. Démarrage :
`cd aiotrade && docker compose up --build`, puis <http://localhost:8088>.
