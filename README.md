# AIO v3 – Gödel Machine + 12 Features

## Features
| # | Feature | Module |
|---|---------|--------|
| 1 | Cache sémantique cosine | `aio/runtime/cache.py` |
| 2 | Streaming natif tokens | `aio/runtime/client.py` |
| 3 | Circuit breaker multi-provider | `aio/runtime/client.py` |
| 4 | Sélecteur modèle par complexité | `aio/runtime/client.py` |
| 5 | Prompt shield (guardrails) | `aio/runtime/guardrails.py` |
| 6 | Validation sortie JSON + retry | `aio/runtime/guardrails.py` |
| 7 | Mémoire persistante SQLite | `aio/runtime/memory.py` |
| 8 | Tool use (web, python, file) | `aio/runtime/client.py` |
| 9 | Multi-tenancy + quotas | `aio/tenancy/manager.py` |
| 10 | Dashboard temps réel | `dashboard/index.html` |
| 11 | OpenTelemetry / Prometheus | `aio/telemetry/otel.py` |
| 12 | Rollback automatique mutations | `aio/compiler/rollback_manager.py` |

## Démarrage
```bash
pip install -r requirements.txt
cp .env.example .env
pytest tests/ -v          # 63 passed
python -m aio.api.main    # http://localhost:8000
```

## Routes clés
- `POST /v1/query` – Query avec cache + shield + model selector
- `POST /v1/stream` – Streaming SSE tokens
- `GET  /dashboard` – Dashboard temps réel
- `GET  /metrics` – Prometheus `/metrics`
- `POST /admin/tenants` – Créer tenant + API key
- `GET  /agents/{id}/memory` – Historique agent
- `GET  /mutations` – État rollback mutations
- `GET  /events/telemetry` – SSE 60/20/20
- `WS   /ws/telemetry` – WebSocket live
