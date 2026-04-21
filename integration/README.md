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
│   ├── context_deviation.py  # pgvector-backed session/turn deviation detection
│   ├── ensemble.py           # Blends all classifier outputs
│   ├── exfiltration.py
│   ├── injection.py
│   ├── obfuscation.py
│   └── url_risk.py
├── configs/
│   ├── classifiers.yaml      # Embedding model, RAG config, per-classifier thresholds
│   ├── fraud_patterns.yaml   # Seed knowledge base (27 labelled fraud patterns)
│   ├── risk_weights.yaml     # Per-parameter risk weights
│   ├── system_prompt.txt     # Bundled LLM system prompt (kept in sync with llm/prompts/)
│   └── thresholds.yaml       # Decision thresholds
├── hitl/
│   ├── migrations/
│   │   ├── 0001_create_hitl_and_audit.py
│   │   └── 0002_add_pgvector_tables.py  # vector extension + session/pattern tables
│   └── queue.py           # PostgreSQL-backed HITL queue
├── output_validator/
│   └── validator.py       # PII and system-prompt-leak detector
├── policies/
│   └── enforcer.py        # Blocked tool-call policy engine
├── risk_engine/
│   └── aggregator.py      # Weighted risk score aggregation
├── sanitizer/
│   └── sanitizer.py       # Input normalization and length enforcement
├── vector_store/
│   ├── encoder.py         # Shared SentenceTransformer singleton (lazy-loaded)
│   ├── fraud_patterns.py  # Knowledge base seeding + RAG retrieval + context formatter
│   └── store.py           # pgvector CRUD: session embeddings, ANN search, drift detection
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

| Tool           | Version                  |
| -------------- | ------------------------ |
| Python         | 3.11+                    |
| Docker         | 24+                      |
| Docker Compose | v2+                      |
| PostgreSQL     | 16+pgvector (via Docker) |
| Redis          | 7 (via Docker)           |

---

## Local Setup (Docker)

### 1 — Configure environment

```bash
cd integration
cp .env.example .env
# Edit .env — at minimum set API_KEYS and JWT_SECRET_KEY
```

**Required env vars:**

| Variable                 | Default                                                          | Description                                    |
| ------------------------ | ---------------------------------------------------------------- | ---------------------------------------------- |
| `DATABASE_URL`           | `postgresql+asyncpg://fraud:fraud@postgres:5432/fraud_detection` | Async Postgres DSN                             |
| `REDIS_URL`              | `redis://redis:6379/0`                                           | Redis DSN                                      |
| `LLM_SERVER_URL`         | `http://localhost:8001`                                          | LLM inference server                           |
| `LLM_ENDPOINT`           | `/v1/chat/completions`                                           | LLM inference endpoint path                    |
| `LLM_MODEL_NAME`         | `fraud-detector-v1`                                              | Model name in request body                     |
| `LLM_SYSTEM_PROMPT_PATH` | _(empty)_                                                        | Path to system prompt; empty = bundled default |
| `RAG_ENABLED`            | `true`                                                           | Enable fraud pattern RAG retrieval             |
| `API_KEYS`               | _(none)_                                                         | Comma-separated valid API keys                 |
| `JWT_SECRET_KEY`         | `change-me-in-production`                                        | **Must change in prod**                        |
| `RISK_ALLOW_THRESHOLD`   | `0.3`                                                            | Score below which → allow                      |
| `RISK_REVIEW_THRESHOLD`  | `0.7`                                                            | Score above which → block                      |
| `RATE_LIMIT_PER_MINUTE`  | `100`                                                            | Per API-key request cap                        |

### 2 — Start all services

```bash
docker compose up --build
```

Services started:

- `api` — FastAPI on port **8000**
- `postgres` — PostgreSQL 16 with **pgvector** extension on port **5432**
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

## Vector Store & RAG

### Overview

The integration layer uses **pgvector** (bundled in the `pgvector/pgvector:pg16` Docker image) for two purposes:

| Purpose                      | Table                | Description                                                                                                                                               |
| ---------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Session embedding store      | `session_embeddings` | Stores per-turn text embeddings for in-session context deviation detection. Rows expire via `expires_at` column (TTL configurable in `classifiers.yaml`). |
| Fraud pattern knowledge base | `fraud_patterns`     | 27 seed examples of known fraud patterns, embedded at startup. Used for RAG retrieval at inference time.                                                  |

Both tables carry `vector(768)` columns with **IVFFlat cosine** indexes (created by migration `0002`).

### Embedding model

All embeddings use **`all-mpnet-base-v2`** (768-dim), loaded once per process as a lazy singleton in `vector_store/encoder.py`. The model ID and embedding dimension are configurable in `configs/classifiers.yaml` under the `embeddings:` section.

### RAG pipeline (per request)

```
Input text
   │
   ▼ embed_text()                     (shared, computed once per request)
   │
   ├─► search_fraud_patterns()        (ANN cosine search, top-K ≥ similarity threshold)
   │        │
   │        ▼ format_rag_context()    → [CONTEXT: ...] reference block
   │
   └─► _call_llm(rag_context=...)     → injected as prefix in user message
```

The RAG context block is clearly labelled so the LLM knows it is reference material only (see Rule 9 in `configs/system_prompt.txt`). The block is token-capped (default 500 tokens) to avoid crowding the actual input.

### Cross-session drift detection

`context_deviation.py` uses `find_similar_recent_sessions()` (signal 6) to detect coordinated attacks: if ≥ 3 distinct sessions send near-identical content within a configurable lookback window (default 5 minutes), the `context_deviation` score is escalated and a `cross_session_coordination:N` flag is added to the result.

### Configuration (`configs/classifiers.yaml`)

```yaml
embeddings:
  model_id: "all-mpnet-base-v2"
  dim: 768
  session_ttl_seconds: 3600
  turn_history_max: 20

rag:
  enabled: true
  top_k: 3
  min_similarity_threshold: 0.65
  max_context_tokens: 500
  seed_on_startup: true
```

### Knowledge base re-seeding

The knowledge base seeds automatically on first startup if the `fraud_patterns` table is empty and `rag.seed_on_startup: true`. To force a re-seed after editing `configs/fraud_patterns.yaml`:

```bash
# Truncate the table, then restart the API service
docker compose exec postgres psql -U fraud fraud_detection \
  -c "TRUNCATE fraud_patterns;"
docker compose restart api
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
