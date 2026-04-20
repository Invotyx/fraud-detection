# Fraud Detection – 2-Week Delivery Timeline

**Infrastructure:** AWS g4dn.xlarge (NVIDIA T4 16GB, 4 vCPU, 16GB RAM)  
**Approach:** LLM training and Integration Layer built in parallel from Day 1

---

## Parallel Tracks

| Track           | Owner                  | Focus                                                |
| --------------- | ---------------------- | ---------------------------------------------------- |
| **LLM**         | `todo_llm.md`          | Fine-tuning, resilient behavior, structured output   |
| **Integration** | `todo_integrations.md` | Sanitization, classifiers, risk scoring, HITL, audit |

---

## Week 1 — Foundation & Core Build

### Day 1 — Environment Bootstrap (Both Tracks)

| LLM Track                                                                            | Integration Track                                            | Date         |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ------------ |
| Install CUDA 12, Python 3.11, PyTorch, Hugging Face Transformers, PEFT, bitsandbytes | Bootstrap FastAPI project, Docker Compose, PostgreSQL, Redis | Apr 16 (Thu) |
| Configure QLoRA 4-bit on T4 (memory-efficient fine-tuning)                           | Set up structured logging (audit trail foundation)           | Apr 16 (Thu) |
| Pull base model (Mistral-7B or Phi-3-mini)                                           | Define API contracts between LLM and Integration Layer       | Apr 16 (Thu) |

---

### Day 2 — Data Preparation + Architecture Design

| LLM Track                                                         | Integration Track                                                                       | Date         |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------ |
| Curate / generate labeled dataset covering all 7 fraud parameters | Design Integration Layer pipeline: Input → Sanitize → Classify → Score → Output         | Apr 17 (Fri) |
| Label examples: benign vs. fraudulent for each parameter          | Set up input sanitization module (HTML parser, script stripper, hidden element remover) | Apr 17 (Fri) |
| Structure prompt templates (system prompt + few-shot examples)    | Define risk scoring schema (per-parameter weight, unified score)                        | Apr 17 (Fri) |

---

### Day 3 — Fine-Tuning Run 1 + Classifiers Start

| LLM Track                                                                                               | Integration Track                                                                 | Date         |
| ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------ |
| Launch fine-tuning run 1: instruction resistance + task adherence (safe behavior on adversarial inputs) | Build URL/Domain Risk analyzer (WHOIS age, reputation blocklist, entropy scoring) | Apr 20 (Mon) |
| Train safe HTML → clean text transformation behavior                                                    | Build Prompt Injection classifier (rule-based + lightweight ML, e.g., DistilBERT) | Apr 20 (Mon) |
| Monitor GPU utilization, adjust batch size / gradient accumulation                                      | Unit test sanitization + URL modules                                              | Apr 20 (Mon) |

---

### Day 4 — Training Continues + Obfuscation/Exfiltration Modules

| LLM Track                                                              | Integration Track                                                                        | Date         |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------ |
| Fine-tuning run 1 completes; evaluate on held-out adversarial set      | Obfuscation/Evasion detection (base64, hex, Unicode lookalikes, steganographic patterns) | Apr 21 (Tue) |
| Checkpoint best weights                                                | Data Exfiltration detection (regex patterns, volume anomaly, sensitive field scanning)   | Apr 21 (Tue) |
| Begin crafting fraud intent + explanation generation training examples | Context Deviation enforcement module (session context comparator)                        | Apr 21 (Tue) |

---

### Day 5 — Fine-Tuning Run 2 + Risk Scoring Engine

| LLM Track                                                                                                 | Integration Track                                                                      | Date         |
| --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------ |
| Launch fine-tuning run 2: fraud intent detection (semantic), contextual reasoning, explanation generation | Risk Scoring aggregation engine: weighted sum across 7 parameters → unified risk score | Apr 22 (Wed) |
| Train structured JSON output format (scores + explanations per parameter)                                 | Action/Tool restriction policy engine (allowlist/denylist, scope enforcement)          | Apr 22 (Wed) |
| Intermediate evaluation: precision/recall on fraud intent examples                                        | Unit test all classifiers end-to-end                                                   | Apr 22 (Wed) |

---

### Day 6 — Mid-Week Evaluation & Adjustments

| LLM Track                                                            | Integration Track                                                                  | Date         |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------ |
| Evaluate run 2 checkpoint; adjust learning rate / data mix if needed | Fraud model ensemble integration (connect existing models as parallel signals)     | Apr 23 (Thu) |
| Test content prioritization: visible vs. hidden content handling     | Output Validation module: PII detection (presidio or regex), unsafe content filter | Apr 23 (Thu) |
| Identify failure modes on 7-parameter benchmark                      | Integration smoke tests (API routes, middleware chain)                             | Apr 23 (Thu) |

---

### Day 7 — Buffer / Catch-up Day _(Apr 24, Fri)_

- Resolve any training instability or classifier underperformance
- Align on JSON output schema shared between LLM and Integration Layer
- Code review both tracks

---

## Week 2 — Integration, Testing & Hardening

### Day 8 — Pipeline Assembly

| LLM Track                                                                      | Integration Track                                                                  | Date         |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ------------ |
| Fine-tuning run 3 (if needed): reinforce weak parameters from Day 6 evaluation | Wire full pipeline: Sanitize → Inject Detect → LLM → Output Validate → Risk Score  | Apr 27 (Mon) |
| Export model to optimized format (GGUF or vLLM-compatible)                     | Human-in-the-Loop (HITL) workflow: flag high-risk cases for manual review queue    | Apr 27 (Mon) |
| Test model serving latency on T4 with vLLM or llama.cpp                        | Audit logging: immutable log per request (input hash, scores, decision, timestamp) | Apr 27 (Mon) |

---

### Day 9 — End-to-End Integration Testing

| LLM Track                                                                  | Integration Track                                                   | Date         |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------ |
| Connect LLM inference server to Integration Layer API                      | End-to-end test: benign inputs flow cleanly through pipeline        | Apr 28 (Tue) |
| Validate structured JSON output parsing in Integration Layer               | End-to-end test: adversarial inputs are caught and scored correctly | Apr 28 (Tue) |
| Test instruction-resistance: embedded malicious instructions in HTML input | Test HITL queue: high-risk cases routed correctly                   | Apr 28 (Tue) |

---

### Day 10 — Adversarial Red Team Testing _(Apr 29, Wed)_

Both tracks collaborate:

- Test all 7 fraud parameters with crafted attack payloads
- Prompt injection attempts (direct + indirect)
- Obfuscated payloads (base64, Unicode tricks)
- URL with high-entropy domains + lookalike patterns
- Social engineering in natural language
- Data exfiltration via output manipulation attempts
- Record false positive / false negative rates per parameter

---

### Day 11 — Tuning & Hardening

| LLM Track                                                        | Integration Track                                                    | Date         |
| ---------------------------------------------------------------- | -------------------------------------------------------------------- | ------------ |
| Prompt-tune system prompt based on red team findings             | Adjust classifier thresholds and risk weights based on red team data | Apr 30 (Thu) |
| Retrain on newly discovered failure cases (targeted fine-tuning) | Harden sanitization against new evasion patterns found in Day 10     | Apr 30 (Thu) |

---

### Day 12 — Performance & Reliability

| LLM Track                                           | Integration Track                                         | Date        |
| --------------------------------------------------- | --------------------------------------------------------- | ----------- |
| Benchmark inference latency (target < 2s p95 on T4) | Load test API pipeline (concurrent requests, queue depth) | May 1 (Fri) |
| Quantization tuning if latency target not met       | Ensure Redis cache for URL reputation lookups             | May 1 (Fri) |
| Final model export + versioning                     | Rate limiting, circuit breakers on external lookups       | May 1 (Fri) |

---

### Day 13 — Final QA & Deployment Prep _(May 4, Mon)_

- Full regression test suite run
- Validate audit logs are complete and tamper-evident
- Review HITL queue workflow with stakeholders
- Deploy to g4dn with systemd / Docker for process management
- Set up CloudWatch alerts (GPU utilization, error rates, latency)

---

### Day 14 — Go-Live & Monitoring _(May 5, Tue)_

- Production deployment
- Validate live traffic flows through full pipeline
- Monitor first 24h: risk score distribution, HITL queue volume, false positive rate
- Handoff documentation

---

## Risk Flags

| Risk                                                | Mitigation                                                              |
| --------------------------------------------------- | ----------------------------------------------------------------------- |
| T4 16GB too constrained for 7B model fine-tuning    | Use QLoRA 4-bit + gradient checkpointing; fallback to Phi-3-mini (3.8B) |
| Dataset too small for reliable fine-tuning          | Augment with synthetic adversarial examples (GPT-4 generated)           |
| Integration classifiers underperform on obfuscation | Add rule-based fallback layer alongside ML classifier                   |
| Pipeline latency exceeds target                     | Cache URL lookups, pre-sanitize async, batch LLM requests               |
