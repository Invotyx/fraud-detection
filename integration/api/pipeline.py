"""
Pipeline Assembler — Phase 13
Full async request pipeline wiring all integration modules together.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncConnection

from integration.api.config import get_settings
from integration.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    Decision,
    FraudAnalysisResult,
    ParameterScore,
)
from integration.audit.logger import log_request, log_stage
from integration.classifiers.context_deviation import check_context_deviation
from integration.classifiers.authority_spoof import detect_authority_spoof
from integration.classifiers.ensemble import EnsembleResult, run_ensemble
from integration.classifiers.exfiltration import detect_exfiltration
from integration.classifiers.obfuscation import detect_obfuscation
from integration.hitl.queue import enqueue as hitl_enqueue
from integration.output_validator.validator import validate_output
from integration.policies.enforcer import check_tool_allowed, scan_llm_output_for_tool_calls
from integration.risk_engine.aggregator import (
    AggregationResult,
    HardOverrideFlags,
    aggregate,
)
from integration.sanitizer.sanitizer import sanitize
from integration.classifiers.injection import classify_injection, scan_for_indirect_injection, classify_injection_from_urls
from integration.classifiers.session_risk import accumulate_session_injection
from integration.vector_store.encoder import encode as embed_text
from integration.vector_store.fraud_patterns import format_rag_context, retrieve_similar_patterns
from integration.vector_store.store import ensure_vector_codec

# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_CACHE: Optional[str] = None


def _load_system_prompt() -> str:
    """Load the LLM system prompt from the configured path (cached after first load)."""
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE

    settings = get_settings()
    path = settings.llm_system_prompt_path

    if not path:
        # Default: bundled prompt next to this package
        import os
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "system_prompt.txt"
        )

    try:
        with open(path) as fh:
            _SYSTEM_PROMPT_CACHE = fh.read().strip()
    except OSError:
        # Fallback minimal prompt — safe operation even without the file
        _SYSTEM_PROMPT_CACHE = (
            "You are a fraud detection AI. Analyze the input and return a "
            "JSON risk assessment with scores for each fraud parameter."
        )

    return _SYSTEM_PROMPT_CACHE


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


async def _call_llm(
    sanitized_text: str,
    *,
    timeout: float,
    llm_url: str,
    llm_endpoint: str,
    model_name: str,
    rag_context: str = "",
) -> Optional[Dict[str, Any]]:
    """
    POST to the LLM inference server (OpenAI-compatible /v1/chat/completions).

    When *rag_context* is provided it is prepended to the user message as a
    clearly-labelled reference block so the model can use it for few-shot
    context without being misled by the examples.
    """
    try:
        import httpx

        user_content = (
            f"<external_data>\n{rag_context}\n</external_data>\n---\nINPUT TO ANALYZE:\n{sanitized_text}"
            if rag_context
            else sanitized_text
        )
        _cfg = get_settings()
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": _load_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            "temperature": _cfg.llm_temperature,
            "max_tokens": _cfg.llm_max_tokens,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{llm_url}{llm_endpoint}", json=payload)
            resp.raise_for_status()
            data = resp.json()
            # Extract assistant message content from OpenAI response envelope
            import json as _json
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return _json.loads(content) if content else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ML classifier runner
# ---------------------------------------------------------------------------

async def _run_ml_classifiers(
    sanitized_text: str,
    session_id: Optional[str],
    task_scope: Optional[str],
    conn: Any,
) -> Dict[str, float]:
    """
    Run ML classifiers in parallel.

    context_deviation now uses pgvector via *conn* and runs as a native coroutine
    (no asyncio.to_thread needed).  obfuscation / exfiltration remain CPU-bound
    sync functions and are dispatched to the thread pool.
    """
    obfuscation_task = asyncio.create_task(
        asyncio.to_thread(detect_obfuscation, sanitized_text)
    )
    exfiltration_task = asyncio.create_task(
        asyncio.to_thread(detect_exfiltration, sanitized_text)
    )
    authority_spoof_task = asyncio.create_task(
        asyncio.to_thread(detect_authority_spoof, sanitized_text)
    )

    obfuscation_result, exfiltration_result, authority_spoof_result = await asyncio.gather(
        obfuscation_task, exfiltration_task, authority_spoof_task
    )

    scores: Dict[str, float] = {
        "obfuscation_evasion": obfuscation_result.score,
        "data_exfiltration": exfiltration_result.score,
        "authority_spoof": authority_spoof_result.score,
    }

    if session_id:
        deviation = await check_context_deviation(
            sanitized_text, session_id, conn, task_scope
        )
        scores["context_deviation"] = deviation.score
    else:
        scores["context_deviation"] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Pipeline core
# ---------------------------------------------------------------------------


async def run_pipeline(
    request: AnalyzeRequest,
    db: AsyncConnection,
    redis_client: Any = None,
    *,
    trace_id: UUID,
) -> AnalyzeResponse:
    """
    Full fraud detection pipeline.

    Steps
    -----
    1.  Sanitize input
    2.  URL risk (via sanitizer-extracted URLs)
    3.  Injection check (rule-based fast path, already in sanitizer result)
    4.  Obfuscation detection
    [parallel]
    5a. LLM inference call
    5b. ML classifier scores (exfiltration, deviation, obfuscation)
    5c. Fraud ensemble models
    6.  Aggregate all scores → Risk Engine
    7.  Output validation (LLM response)
    8.  Policy enforcement scan
    9.  Decision + HITL routing
    10. Write audit log → return response
    """
    settings = get_settings()
    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # Step 0: Guard model fast path — rule-based injection pre-filter
    # If a definitive hard-block injection pattern is found in the raw
    # input, skip the full pipeline and return an immediate block.
    # ------------------------------------------------------------------
    guard_result = classify_injection(request.content, use_ml=False)
    if guard_result.rule_match and guard_result.score >= settings.guard_prefilter_score_threshold:
        log_stage("guard_prefilter", trace_id=trace_id,
                  score=guard_result.score, flags=guard_result.flags)
        _block_score = guard_result.score
        _guard_result = FraudAnalysisResult(
            prompt_injection=ParameterScore(
                score=_block_score, flag=True,
                reason="Guard pre-filter: definitive injection pattern matched",
            ),
            unified_risk_score=_block_score,
            decision=Decision.BLOCK,
            explanation="Prompt injection blocked by pre-filter.",
        )
        _proc_ms = int((time.monotonic() - t_start) * 1000)
        await log_request(
            db,
            trace_id=trace_id,
            sanitized_text=request.content[:settings.audit_log_content_max_length],
            raw_text=request.content,
            classifier_scores={"prompt_injection": _block_score},
            llm_response=None,
            unified_risk_score=_block_score,
            decision=Decision.BLOCK.value,
            flags={"guard_prefilter": True, "flags": guard_result.flags},
            hitl_required=False,
            processing_time_ms=_proc_ms,
        )
        return AnalyzeResponse(
            trace_id=trace_id,
            result=_guard_result,
            processing_time_ms=_proc_ms,
            hitl_pending=False,
            mitigation_notice=(
                "A prompt injection attack was detected and blocked before reaching the model. "
                f"Matched rule(s): {', '.join(guard_result.flags[:3])}."
            ),
            blocked_attack_type="prompt_injection",
        )

    # ------------------------------------------------------------------
    # Step 1: Sanitize
    # ------------------------------------------------------------------
    sanitized = sanitize(request.content)
    log_stage(
        "sanitize",
        trace_id=trace_id,
        removed=len(sanitized.removed_elements),
        urls_found=len(sanitized.detected_urls),
        anomalies=sanitized.encoding_anomalies,
    )

    # URL query-parameter injection scan (CVE-2026-24307 / Reprompt pattern):
    # URLs extracted by the sanitizer are not re-checked for injection payloads
    # embedded in their query parameters by the guard prefilter above.
    if sanitized.detected_urls:
        url_param_result = classify_injection_from_urls(
            sanitized.detected_urls)
        if url_param_result.rule_match and url_param_result.score >= settings.guard_prefilter_score_threshold:
            log_stage("url_param_injection", trace_id=trace_id,
                      score=url_param_result.score, flags=url_param_result.flags)
            _block_score = url_param_result.score
            _url_result = FraudAnalysisResult(
                prompt_injection=ParameterScore(
                    score=_block_score, flag=True,
                    reason="URL query parameter injection detected",
                ),
                url_domain_risk=ParameterScore(score=0.7, flag=True,
                                               reason="Malicious payload in URL parameter"),
                unified_risk_score=_block_score,
                decision=Decision.BLOCK,
                explanation="Prompt injection payload detected in URL query parameter.",
            )
            _proc_ms = int((time.monotonic() - t_start) * 1000)
            await log_request(
                db,
                trace_id=trace_id,
                sanitized_text=sanitized.sanitized_text[:
                                                        settings.audit_log_content_max_length],
                raw_text=request.content,
                classifier_scores={"prompt_injection": _block_score},
                llm_response=None,
                unified_risk_score=_block_score,
                decision=Decision.BLOCK.value,
                flags={"url_param_injection": True,
                       "flags": url_param_result.flags},
                hitl_required=False,
                processing_time_ms=_proc_ms,
            )
            return AnalyzeResponse(
                trace_id=trace_id,
                result=_url_result,
                processing_time_ms=_proc_ms,
                hitl_pending=False,
                mitigation_notice=(
                    "A prompt injection payload was detected in a URL query parameter "
                    f"and blocked. Matched rule(s): {', '.join(url_param_result.flags[:3])}."
                ),
                blocked_attack_type="prompt_injection",
            )

    # ------------------------------------------------------------------
    # Step 2: Compute shared text embedding (once, reused across all steps)
    # ------------------------------------------------------------------
    text_embedding = await asyncio.to_thread(embed_text, sanitized.sanitized_text)

    # ------------------------------------------------------------------
    # Step 3: RAG — retrieve similar fraud patterns and format context
    # ------------------------------------------------------------------
    rag_context = ""
    if settings.rag_enabled and text_embedding is not None:
        try:
            raw_conn = await db.get_raw_connection()
            rag_driver_conn = raw_conn.driver_connection
            await ensure_vector_codec(rag_driver_conn)
            similar_patterns = await retrieve_similar_patterns(
                rag_driver_conn, text_embedding
            )
            if similar_patterns:
                rag_context = format_rag_context(similar_patterns)
                log_stage(
                    "rag",
                    trace_id=trace_id,
                    patterns_found=len(similar_patterns),
                    top_similarity=similar_patterns[0]["similarity"],
                )
        except Exception:
            pass  # RAG failure is non-fatal — pipeline continues without it

    # ------------------------------------------------------------------
    # Steps 5a / 5b / 5c  (parallel)
    # ------------------------------------------------------------------
    raw_conn_obj = None
    driver_conn = None
    try:
        raw_conn_obj = await db.get_raw_connection()
        driver_conn = raw_conn_obj.driver_connection
        await ensure_vector_codec(driver_conn)
    except Exception:
        driver_conn = None

    llm_task = asyncio.create_task(
        _call_llm(
            sanitized.sanitized_text,
            timeout=float(settings.llm_request_timeout),
            llm_url=settings.llm_server_url,
            llm_endpoint=settings.llm_endpoint,
            model_name=settings.llm_model_name,
            rag_context=rag_context,
        )
    )

    ml_task = asyncio.create_task(
        _run_ml_classifiers(
            sanitized.sanitized_text,
            request.session_id,
            request.task_scope,
            driver_conn,
        )
    )

    ensemble_features: Dict[str, Any] = {"text": sanitized.sanitized_text}
    if text_embedding is not None:
        ensemble_features["embedding"] = text_embedding

    ensemble_task = asyncio.create_task(
        run_ensemble(ensemble_features)
    )

    try:
        remaining = settings.pipeline_timeout_seconds - \
            (time.monotonic() - t_start)
        llm_response_raw, ml_scores, ensemble_result = await asyncio.wait_for(
            asyncio.gather(llm_task, ml_task, ensemble_task),
            timeout=max(remaining, 0.5),
        )
    except asyncio.TimeoutError:
        log_stage("parallel_stages", trace_id=trace_id, status="timeout")
        llm_response_raw = None
        ml_scores = {"obfuscation_evasion": 0.0,
                     "data_exfiltration": 0.0, "context_deviation": 0.0,
                     "authority_spoof": 0.0}
        ensemble_result = None

    ensemble_result: Optional[EnsembleResult]

    log_stage("parallel_stages", trace_id=trace_id,
              llm_ok=(llm_response_raw is not None),
              ensemble_score=getattr(ensemble_result, "combined_score", None))

    # ------------------------------------------------------------------
    # Step 3 + 4: Extract injection / URL risk from sanitized result
    # (Rule-based scores already embedded in encoding_anomalies; we use
    #  presence of anomalies as a proxy score here)
    # ------------------------------------------------------------------
    injection_score = settings.injection_encoding_anomaly_score if sanitized.encoding_anomalies else 0.0
    url_risk_score = 0.0
    if sanitized.detected_urls:
        # Lightweight heuristic: presence of suspicious redirect / shortener
        suspicious_prefixes = (
            "http://bit.ly", "http://t.co", "http://tinyurl")
        url_risk_score = settings.url_risk_suspicious_score if any(
            u.startswith(suspicious_prefixes) for u in sanitized.detected_urls
        ) else settings.url_risk_default_score

    # ------------------------------------------------------------------
    # Step 6: Build parameter scores dict for aggregation
    # ------------------------------------------------------------------
    # LLM per-parameter scores (fall back to 0 if LLM unavailable)
    param_scores: Dict[str, float] = {
        "url_domain_risk": url_risk_score,
        "prompt_injection": injection_score,
        "obfuscation_evasion": ml_scores.get("obfuscation_evasion", 0.0),
        "data_exfiltration": ml_scores.get("data_exfiltration", 0.0),
        "context_deviation": ml_scores.get("context_deviation", 0.0),
        "authority_spoof": ml_scores.get("authority_spoof", 0.0),
    }

    if llm_response_raw:
        for param in ("url_domain_risk", "fraud_intent", "prompt_injection",
                      "context_deviation", "data_exfiltration",
                      "obfuscation_evasion", "unauthorized_action",
                      "authority_spoof"):
            block = llm_response_raw.get(param, {})
            if isinstance(block, dict) and "score" in block:
                # Blend LLM score with rule-based score (weights from settings)
                rule_score = param_scores.get(param, 0.0)
                param_scores[param] = (
                    settings.llm_score_weight * float(block["score"])
                    + settings.rule_score_weight * rule_score
                )
        param_scores.setdefault("fraud_intent", float(
            (llm_response_raw.get("fraud_intent") or {}).get("score", 0.0)
        ))
        param_scores.setdefault("unauthorized_action", float(
            (llm_response_raw.get("unauthorized_action") or {}).get("score", 0.0)
        ))
        param_scores.setdefault("authority_spoof", max(
            param_scores.get("authority_spoof", 0.0),
            float((llm_response_raw.get("authority_spoof") or {}).get("score", 0.0)),
        ))
    else:
        param_scores.setdefault("fraud_intent", 0.0)
        param_scores.setdefault("unauthorized_action", 0.0)

    # Blend ensemble score into fraud_intent if available
    if ensemble_result and ensemble_result.combined_score > 0:
        param_scores["fraud_intent"] = max(
            param_scores["fraud_intent"], ensemble_result.combined_score
        )

    # ------------------------------------------------------------------
    # Step 6: Aggregate
    # ------------------------------------------------------------------
    hard_flags = HardOverrideFlags(
        url_blocklist_match=url_risk_score >= 0.95,
        injection_rule_match=bool(
            sanitized.encoding_anomalies and injection_score >= 0.9),
    )

    agg: AggregationResult = aggregate(param_scores, hard_flags)
    log_stage("aggregate", trace_id=trace_id,
              unified_risk_score=agg.unified_risk_score, decision=agg.decision.value)

    # ------------------------------------------------------------------
    # Step 7: Output validation
    # ------------------------------------------------------------------
    validation_flags: Dict[str, Any] = {}
    if llm_response_raw:
        val_result = validate_output(
            llm_response_raw,
            system_prompt_fragments=None,
        )
        _pii_detected = any("pii_in_output:" in i for i in val_result.issues)
        _prompt_leaked = any(
            "system_prompt_leak:" in i for i in val_result.issues)
        validation_flags = {
            "valid_schema": val_result.valid,
            "pii_detected": _pii_detected,
            "prompt_leak": _prompt_leaked,
            "unsafe_content": any("unsafe_content:" in i for i in val_result.issues),
            "issues": val_result.issues,
        }
        # Escalate risk if output contains PII or prompt leakage
        if _pii_detected or _prompt_leaked:
            agg = AggregationResult(
                parameter_scores=agg.parameter_scores,
                unified_risk_score=max(agg.unified_risk_score, 0.75),
                decision=Decision.BLOCK,
                hard_override=True,
                override_reason="output_validation_fail",
                weights_used=agg.weights_used,
            )

    # ------------------------------------------------------------------
    # Step 8: Policy enforcement (scan LLM output for tool calls)
    # ------------------------------------------------------------------
    policy_violations = []
    if llm_response_raw:
        import json as _json
        enforcement_results = scan_llm_output_for_tool_calls(
            _json.dumps(llm_response_raw))
        policy_violations = [
            r.reason for r in enforcement_results if not r.allowed
        ]
        if policy_violations:
            log_stage("policy", trace_id=trace_id,
                      blocked_tools=policy_violations)
            agg = AggregationResult(
                parameter_scores=agg.parameter_scores,
                unified_risk_score=max(agg.unified_risk_score, 0.85),
                decision=Decision.BLOCK,
                hard_override=True,
                override_reason="policy_violation",
                weights_used=agg.weights_used,
            )

    # ------------------------------------------------------------------
    # Step 9: HITL routing
    # ------------------------------------------------------------------
    hitl_pending = False
    if agg.decision == Decision.REVIEW:
        await hitl_enqueue(
            db,
            request_id=str(trace_id),
            unified_risk_score=agg.unified_risk_score,
            classifier_scores=param_scores,
            llm_response=llm_response_raw,
        )
        hitl_pending = True
        log_stage("hitl", trace_id=trace_id, enqueued=True)

    # ------------------------------------------------------------------
    # Step 9b: Session-level injection accumulation (INT-7)
    # Detects payload-splitting across multiple moderate-risk requests.
    # ------------------------------------------------------------------
    session_result = await accumulate_session_injection(
        redis_client,
        session_id=request.session_id or "",
        injection_score=param_scores.get("prompt_injection", 0.0),
    )
    if session_result.escalated and agg.decision != Decision.BLOCK:
        log_stage("session_risk", trace_id=trace_id,
                  accumulated=session_result.accumulated_score,
                  count=session_result.request_count)
        agg = AggregationResult(
            parameter_scores=agg.parameter_scores,
            unified_risk_score=max(agg.unified_risk_score, 0.75),
            decision=Decision.BLOCK,
            hard_override=True,
            override_reason="session_injection_accumulation",
            weights_used=agg.weights_used,
        )

    # ------------------------------------------------------------------
    # Step 10: Write audit log
    # ------------------------------------------------------------------
    processing_time_ms = int((time.monotonic() - t_start) * 1000)

    flags_payload = {
        **(validation_flags),
        "policy_violations": policy_violations,
        "hard_override": agg.hard_override,
        "override_reason": agg.override_reason,
    }

    try:
        await log_request(
            db,
            trace_id=trace_id,
            sanitized_text=sanitized.sanitized_text,
            raw_text=request.content,
            classifier_scores=param_scores,
            llm_response=llm_response_raw,
            unified_risk_score=agg.unified_risk_score,
            decision=agg.decision.value,
            flags=flags_payload,
            hitl_required=hitl_pending,
            processing_time_ms=processing_time_ms,
        )
    except Exception as exc:
        log_stage(
            "audit_write_failed",
            trace_id=trace_id,
            error=str(exc),
        )

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    def _ps(key: str) -> ParameterScore:
        s = param_scores.get(key, 0.0)
        llm_block = (llm_response_raw or {}).get(key, {})
        reason = llm_block.get("reason", "") if isinstance(llm_block, dict) else ""
        return ParameterScore(score=s, flag=s >= 0.7, reason=reason)

    result = FraudAnalysisResult(
        url_domain_risk=_ps("url_domain_risk"),
        fraud_intent=_ps("fraud_intent"),
        prompt_injection=_ps("prompt_injection"),
        context_deviation=_ps("context_deviation"),
        data_exfiltration=_ps("data_exfiltration"),
        obfuscation_evasion=_ps("obfuscation_evasion"),
        unauthorized_action=_ps("unauthorized_action"),
        authority_spoof=_ps("authority_spoof"),
        unified_risk_score=agg.unified_risk_score,
        decision=agg.decision,
        explanation=str(llm_response_raw.get("explanation", "")
                        ) if llm_response_raw else "",
    )

    # INT-1: Build mitigation notice when the request is blocked or flagged.
    mitigation_notice: Optional[str] = None
    blocked_attack_type: Optional[str] = None
    if agg.decision != Decision.ALLOW:
        _attack_scores = {
            k: param_scores.get(k, 0.0)
            for k in (
                "prompt_injection", "data_exfiltration", "obfuscation_evasion",
                "authority_spoof", "unauthorized_action", "fraud_intent",
                "url_domain_risk", "context_deviation",
            )
        }
        blocked_attack_type = max(_attack_scores, key=_attack_scores.get)
        _action = "blocked" if agg.decision == Decision.BLOCK else "flagged for review"
        _readable = blocked_attack_type.replace("_", " ")
        mitigation_notice = (
            f"Request {_action}. Dominant risk signal: {_readable} "
            f"(score {_attack_scores[blocked_attack_type]:.2f}). "
            "No action was taken on the original content."
        )

    return AnalyzeResponse(
        trace_id=trace_id,
        result=result,
        processing_time_ms=processing_time_ms,
        hitl_pending=hitl_pending,
        mitigation_notice=mitigation_notice,
        blocked_attack_type=blocked_attack_type,
    )
