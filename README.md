# Fraud Detection System

A two-track system for detecting fraud signals in text inputs:

- **Integration layer** — FastAPI service that runs a real-time multi-classifier pipeline (rule-based + ML + LLM ensemble) and exposes a REST API.
- **LLM track** — QLoRA fine-tuning pipeline for Mistral-7B-Instruct that produces a structured 7-parameter fraud classifier, hosted as an OpenAI-compatible inference server.

---

## Repository Layout

```
fraud-detection/
├── integration/          # FastAPI service, classifiers, risk engine, audit
│   ├── api/              # FastAPI app, pipeline orchestrator, schemas
│   ├── audit/            # Append-only audit logger (SHA-256 hashing only)
│   ├── classifiers/      # Rule-based signal detectors + ensemble
│   ├── configs/          # thresholds.yaml
│   ├── hitl/             # Human-in-the-loop queue + DB migrations
│   ├── output_validator/ # PII / system-prompt-leak validator
│   ├── policies/         # Blocked-tool policy enforcer
│   ├── risk_engine/      # Score aggregation
│   ├── sanitizer/        # Input sanitizer
│   ├── scripts/          # load_test.py
│   ├── tests/
│   │   ├── unit/         # 13 unit test modules
│   │   └── integration/  # End-to-end adversarial tests
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── requirements.txt
│
└── llm/                  # LLM fine-tuning and serving track
    ├── configs/          # base_config.yaml, run1/run2 overrides
    ├── data/             # Generated JSONL datasets (train/val/test)
    ├── checkpoints/      # LoRA adapters saved here during training
    ├── prompts/          # system_prompt.txt
    ├── scripts/          # All training, eval, and serving scripts
    └── MODEL_CARD.md     # Model documentation
```

---

## The 7 Fraud Parameters

| Parameter           | Weight | Description                                 |
| ------------------- | ------ | ------------------------------------------- |
| url_domain_risk     | 0.15   | Malicious / spoofed URL patterns            |
| fraud_intent        | 0.20   | Social engineering, scam, phishing language |
| prompt_injection    | 0.20   | Attempts to override model instructions     |
| context_deviation   | 0.10   | Task switching outside the defined scope    |
| data_exfiltration   | 0.15   | Attempts to extract secrets or PII          |
| obfuscation_evasion | 0.10   | Base64, homoglyphs, zero-width characters   |
| unauthorized_action | 0.10   | Tool calls or system actions not permitted  |

### Decision thresholds

| Decision | Condition                                      |
| -------- | ---------------------------------------------- |
| allow    | unified_risk_score < 0.30                      |
| review   | 0.30 ≤ unified_risk_score < 0.70               |
| block    | unified_risk_score ≥ 0.70                      |
| block\*  | Any single parameter score > 0.90 (hard block) |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- (LLM track) NVIDIA GPU with 16 GB VRAM (AWS g4dn.xlarge or equivalent)

### 1 — Clone

```bash
git clone <repo-url> fraud-detection
cd fraud-detection
```

### 2 — Start the integration service

```bash
cd integration
docker compose up --build
```

API available at `http://localhost:8000`. See [integration/README.md](integration/README.md).

### 3 — Train and serve the LLM

```bash
cd llm
bash scripts/setup.sh              # one-time: CUDA + Python venv
source ~/venv/bin/activate
python scripts/prepare_data.py     # generate training data
python scripts/train_run1.py       # Run 1
python scripts/train_run2.py       # Run 2
python scripts/merge_and_serve.py  # merge + serve on :8001
```

See [llm/README.md](llm/README.md) for the full workflow.

---

## Architecture

```
User Request
     │
     ▼
FastAPI (/analyze)
     │  X-API-Key auth, rate limiting, X-Trace-ID
     ▼
Pipeline (integration/api/pipeline.py)
     │
     ├─► Sanitizer          (strip control chars, length limit)
     │
     ├─► [parallel]
     │     ├─► LLM server   :8001  →  7 parameter scores
     │     ├─► ML classifiers (obfuscation · exfiltration · deviation)
     │     └─► Rule-based ensemble
     │
     ├─► Aggregator         (60 % LLM + 40 % rule blend)
     │
     ├─► Output validator   (PII / prompt-leak guard → escalate)
     │
     ├─► Policy enforcer    (blocked tool calls → escalate)
     │
     ├─► HITL queue         (REVIEW decisions → Postgres queue)
     │
     └─► Audit logger       (SHA-256 hashes only, append-only)
          │
          ▼
     AnalyzeResponse  {decision, unified_risk_score, parameters, trace_id}
```

---

## Development

See each sub-directory README for module-specific instructions:

- [integration/README.md](integration/README.md) — API setup, env vars, testing, migrations
- [llm/README.md](llm/README.md) — Training workflow, eval, red-team, hardening
