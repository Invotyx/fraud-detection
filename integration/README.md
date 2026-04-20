# Integration Layer

FastAPI service that orchestrates a real-time fraud-detection pipeline combining rule-based classifiers, ML signals, and an LLM ensemble. Backed by PostgreSQL (asyncpg) and Redis.

---

## Directory Structure

```
integration/
├── api/
│   ├── config.py          # Pydantic-settings config (env vars)
│   ├── main.py            # FastAPI app, routes, auth, rate limiting
│   ├── pipeline.py        # Full async pipeline orchestrator
│   └── schemas.py         # Request / Response Pydantic models
├── audit/
│   └── logger.py          # Append-only audit trail (SHA-256, no plaintext)
├── classifiers/
│   ├── context_deviation.py
│   ├── ensemble.py        # Blends all classifier outputs
│   ├── exfiltration.py
│   ├── injection.py
│   ├── obfuscation.py
│   └── url_risk.py
├── configs/
│   └── thresholds.yaml    # Risk score thresholds and parameter weights
├── hitl/
│   ├── migrations/        # Alembic migration: 0001_create_hitl_and_audit.py
│   └── queue.py           # PostgreSQL-backed HITL queue
├── output_validator/
│   └── validator.py       # PII and system-prompt-leak detector
├── policies/
│   └── enforcer.py        # Blocked tool-call policy engine
├── risk_engine/
│   └── aggregator.py      # Weighted risk score aggregation
├── sanitizer/
│   └── sanitizer.py       # Input normalization and length enforcement
├── scripts/
│   └── load_test.py       # Async SLA load tester
├── tests/
│   ├── unit/              # 13 pytest modules (SQLite in-memory, mocked Redis)
│   └── integration/
│       └── test_e2e.py    # 20 end-to-end adversarial tests
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Prerequisites

| Tool           | Version         |
| -------------- | --------------- |
| Python         | 3.11+           |
| Docker         | 24+             |
| Docker Compose | v2+             |
| PostgreSQL     | 16 (via Docker) |
| Redis          | 7 (via Docker)  |

---

## Local Setup (Docker)

### 1 — Configure environment

```bash
cd integration
cp .env.example .env
# Edit .env — at minimum set API_KEYS and JWT_SECRET_KEY
```

**Required env vars:**

| Variable                | Default                                                          | Description                    |
| ----------------------- | ---------------------------------------------------------------- | ------------------------------ |
| `DATABASE_URL`          | `postgresql+asyncpg://fraud:fraud@postgres:5432/fraud_detection` | Async Postgres DSN             |
| `REDIS_URL`             | `redis://redis:6379/0`                                           | Redis DSN                      |
| `LLM_SERVER_URL`        | `http://localhost:8001`                                          | LLM inference server           |
| `API_KEYS`              | _(none)_                                                         | Comma-separated valid API keys |
| `JWT_SECRET_KEY`        | `change-me-in-production`                                        | **Must change in prod**        |
| `RISK_ALLOW_THRESHOLD`  | `0.3`                                                            | Score below which → allow      |
| `RISK_REVIEW_THRESHOLD` | `0.7`                                                            | Score above which → block      |
| `RATE_LIMIT_PER_MINUTE` | `100`                                                            | Per API-key request cap        |

### 2 — Start all services

```bash
docker compose up --build
```

Services started:

- `api` — FastAPI on port **8000**
- `postgres` — PostgreSQL on port **5432**
- `redis` — Redis on port **6379**

### 3 — Run database migrations

```bash
docker compose exec api alembic upgrade head
```

### 4 — Verify

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "..."}
```

---

## API Reference

All endpoints require `X-API-Key` header. An optional `X-Trace-ID` header is
forwarded and returned; auto-generated (UUID4) if absent.

### `POST /analyze`

Analyze text for fraud signals.

**Request:**

```json
{
  "content": "URGENT: verify your account at http://paypal-secure.xyz",
  "session_id": "optional-session-uuid",
  "context": "optional prior turn text"
}
```

**Response:**

```json
{
  "trace_id": "550e8400-...",
  "decision": "block",
  "unified_risk_score": 0.87,
  "parameters": {
    "url_domain_risk": { "score": 0.91, "flag": true, "reason": "..." },
    "fraud_intent": { "score": 0.84, "flag": true, "reason": "..." },
    "prompt_injection": { "score": 0.02, "flag": false, "reason": "" },
    "context_deviation": { "score": 0.05, "flag": false, "reason": "" },
    "data_exfiltration": { "score": 0.03, "flag": false, "reason": "" },
    "obfuscation_evasion": { "score": 0.11, "flag": false, "reason": "" },
    "unauthorized_action": { "score": 0.01, "flag": false, "reason": "" }
  },
  "hitl_required": false,
  "processing_time_ms": 312
}
```

**Decision values:** `allow` · `review` · `block`

---

### `GET /health`

Returns service health. No auth required.

---

### `GET /hitl/queue`

List pending HITL review items. Requires HITL API key (`X-HITL-Key` header).

### `GET /hitl/{id}`

Get a single HITL item by UUID.

### `POST /hitl/{id}/decision`

Submit a human decision for a queued item.

```json
{ "decision": "allow", "reviewer": "analyst@example.com", "notes": "..." }
```

---

## Running Tests

```bash
# Unit tests (no external services needed)
cd integration
pip install -r requirements.txt
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=. --cov-report=term-missing

# End-to-end adversarial tests (requires running services)
pytest tests/integration/ -v
```

---

## Load Testing

```bash
python scripts/load_test.py \
  --url http://localhost:8000 \
  --api-key <your-key> \
  --requests 500 \
  --concurrency 20 \
  --adversarial \
  --output-json results.json
```

SLA checks (exit 1 if any fail):

- Error rate < 1%
- Zero 5xx responses
- p95 latency < 2 000 ms
- p99 latency < 5 000 ms

---

## Production Deployment

### Environment hardening checklist

- [ ] Set strong random `JWT_SECRET_KEY`
- [ ] Set `API_KEYS` to rotating secret values
- [ ] Set `APP_ENV=production`
- [ ] Set `LOG_LEVEL=WARNING`
- [ ] Use managed PostgreSQL (RDS) and ElastiCache (Redis) — update `DATABASE_URL` / `REDIS_URL`
- [ ] Set `LLM_SERVER_URL` to the internal address of the LLM server (see `llm/README.md`)
- [ ] Run behind a TLS-terminating reverse proxy (nginx / ALB)
- [ ] Restrict inbound traffic to API port only

### Docker Compose (production override)

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Kubernetes

The `api` service is stateless and horizontally scalable. Use:

- Deployment with ≥ 2 replicas
- `PodDisruptionBudget` with `minAvailable: 1`
- External PostgreSQL and Redis (not in-cluster)
- `HorizontalPodAutoscaler` on CPU/RPS

### Health / readiness probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 15
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```
