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
from integration.classifiers.context_deviation import (
    check_context_deviation,
    store_session_scope,
)
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

# ---------------------------------------------------------------------------
# LLM client stub — replaced by real HTTP call in production
# ---------------------------------------------------------------------------


async def _call_llm(
    sanitized_text: str,
    *,
    timeout: float,
    llm_url: str,
) -> Optional[Dict[str, Any]]:
    """
    POST sanitized text to the LLM inference server and return the parsed
    JSON response, or None if the call fails / times out.
    """
    try:
        import httpx

        payload = {"content": sanitized_text}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{llm_url}/analyze", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ML classifier runner stub
# ---------------------------------------------------------------------------

async def _run_ml_classifiers(
    sanitized_text: str,
    session_id: Optional[str],
    task_scope: Optional[str],
) -> Dict[str, float]:
    """
    Run remaining ML classifiers in parallel.
    Returns a dict of {parameter_name: score}.
    """
    obfuscation_task = asyncio.create_task(
        asyncio.to_thread(detect_obfuscation, sanitized_text)
    )
    exfiltration_task = asyncio.create_task(
        asyncio.to_thread(detect_exfiltration, sanitized_text)
    )

    obfuscation_result, exfiltration_result = await asyncio.gather(
        obfuscation_task, exfiltration_task
    )

    scores: Dict[str, float] = {
        "obfuscation_evasion": obfuscation_result.score,
        "data_exfiltration": exfiltration_result.score,
    }

    if session_id and task_scope:
        deviation = await asyncio.to_thread(
            check_context_deviation, sanitized_text, session_id, task_scope
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

    # ------------------------------------------------------------------
    # Step 2: Session scope storage (for context deviation)
    # ------------------------------------------------------------------
    if request.session_id and request.task_scope:
        await asyncio.to_thread(
            store_session_scope, request.session_id, request.task_scope
        )

    # ------------------------------------------------------------------
    # Steps 5a / 5b / 5c  (parallel)
    # ------------------------------------------------------------------
    llm_task = asyncio.create_task(
        _call_llm(
            sanitized.sanitized_text,
            timeout=float(settings.llm_request_timeout),
            llm_url=settings.llm_server_url,
        )
    )

    ml_task = asyncio.create_task(
        _run_ml_classifiers(
            sanitized.sanitized_text,
            request.session_id,
            request.task_scope,
        )
    )

    ensemble_task = asyncio.create_task(
        run_ensemble({"text": sanitized.sanitized_text})
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
                     "data_exfiltration": 0.0, "context_deviation": 0.0}
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
    }

    if llm_response_raw:
        for param in ("url_domain_risk", "fraud_intent", "prompt_injection",
                      "context_deviation", "data_exfiltration",
                      "obfuscation_evasion", "unauthorized_action"):
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
              unified_score=agg.unified_score, decision=agg.decision.value)

    # ------------------------------------------------------------------
    # Step 7: Output validation
    # ------------------------------------------------------------------
    validation_flags: Dict[str, Any] = {}
    if llm_response_raw:
        import json as _json
        val_result = validate_output(
            _json.dumps(llm_response_raw),
            system_prompt_fragments=None,
        )
        validation_flags = {
            "valid_schema": val_result.valid,
            "pii_detected": val_result.pii_detected,
            "prompt_leak": val_result.system_prompt_leaked,
            "unsafe_content": val_result.unsafe_content_detected,
            "issues": val_result.issues,
        }
        # Escalate risk if output contains PII or prompt leakage
        if val_result.pii_detected or val_result.system_prompt_leaked:
            agg = AggregationResult(
                parameter_scores=agg.parameter_scores,
                unified_score=max(agg.unified_score, 0.75),
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
            r.tool_name for r in enforcement_results if not r.allowed
        ]
        if policy_violations:
            log_stage("policy", trace_id=trace_id,
                      blocked_tools=policy_violations)
            agg = AggregationResult(
                parameter_scores=agg.parameter_scores,
                unified_score=max(agg.unified_score, 0.85),
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
            request_id=trace_id,
            unified_risk_score=agg.unified_score,
            payload={
                "content_hash": sanitized.sanitized_hash,
                "scores": param_scores,
                "flags": validation_flags,
            },
        )
        hitl_pending = True
        log_stage("hitl", trace_id=trace_id, enqueued=True)

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

    await log_request(
        db,
        trace_id=trace_id,
        sanitized_text=sanitized.sanitized_text,
        raw_text=request.content,
        classifier_scores=param_scores,
        llm_response=llm_response_raw,
        unified_risk_score=agg.unified_score,
        decision=agg.decision.value,
        flags=flags_payload,
        hitl_required=hitl_pending,
        processing_time_ms=processing_time_ms,
    )

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    def _ps(key: str) -> ParameterScore:
        s = param_scores.get(key, 0.0)
        return ParameterScore(score=s, flag=s >= 0.7, reason="")

    result = FraudAnalysisResult(
        url_domain_risk=_ps("url_domain_risk"),
        fraud_intent=_ps("fraud_intent"),
        prompt_injection=_ps("prompt_injection"),
        context_deviation=_ps("context_deviation"),
        data_exfiltration=_ps("data_exfiltration"),
        obfuscation_evasion=_ps("obfuscation_evasion"),
        unauthorized_action=_ps("unauthorized_action"),
        unified_risk_score=agg.unified_score,
        decision=agg.decision,
        explanation=str(llm_response_raw.get("explanation", "")
                        ) if llm_response_raw else "",
    )

    return AnalyzeResponse(
        trace_id=trace_id,
        result=result,
        processing_time_ms=processing_time_ms,
        hitl_pending=hitl_pending,
    )
