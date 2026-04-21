"""
Session-Level Injection Risk Accumulator (INT-7)
-------------------------------------------------
Detects payload-splitting attacks where a malicious actor deliberately fragments
an instruction-override payload across multiple requests in the same session so
that each individual request scores below the block threshold.

Strategy
--------
Each request's prompt_injection score is pushed into a per-session Redis sorted
set keyed on timestamp.  After every push the last *WINDOW_SECONDS* of scores
are summed.  If the running total of *moderate* injection scores
(>= ACCUMULATE_THRESHOLD) reaches SESSION_ESCALATE_THRESHOLD the session is
considered compromised and the current request is escalated to block.

The escalation threshold is deliberately low (1.20) so that three moderate-risk
requests (3 × 0.45 = 1.35) within an hour trigger a block — matching the
payload-splitting threat model where no single fragment crosses 0.70.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from api.config import get_settings

# ---------------------------------------------------------------------------
# Fixed implementation constant (not user-tunable)
# ---------------------------------------------------------------------------

_SESSION_KEY_PREFIX = "session_risk:"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SessionRiskResult:
    accumulated_score: float   # sum of moderate injection scores in window
    request_count: int         # number of requests that contributed
    escalated: bool            # True → current request should be escalated to block


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


async def accumulate_session_injection(
    redis_client,
    session_id: str,
    injection_score: float,
) -> SessionRiskResult:
    """
    Record *injection_score* for *session_id* and return the aggregated risk.

    Safe to call with ``redis_client=None`` — returns a no-op result so the
    pipeline never fails when Redis is unavailable.
    """
    if redis_client is None or not session_id:
        return SessionRiskResult(
            accumulated_score=0.0, request_count=0, escalated=False
        )

    _s = get_settings()
    window_seconds = _s.session_risk_window_seconds
    accumulate_threshold = _s.session_risk_accumulate_threshold
    escalate_threshold = _s.session_risk_escalate_threshold

    key = f"{_SESSION_KEY_PREFIX}{session_id}"
    now = time.time()
    # Each sorted-set member encodes both the timestamp and the score so we can
    # retrieve the score without a second round-trip.
    member = f"{now:.6f}:{injection_score:.6f}"

    try:
        pipe = redis_client.pipeline()
        # 1. Add this request's score
        pipe.zadd(key, {member: now})
        # 2. Prune entries outside the rolling window
        pipe.zremrangebyscore(key, 0, now - window_seconds)
        # 3. Refresh TTL so idle sessions expire automatically
        pipe.expire(key, int(window_seconds))
        # 4. Fetch all remaining members
        pipe.zrange(key, 0, -1)
        _, _, _, members = await pipe.execute()

        total = 0.0
        count = 0
        for m in members:
            try:
                _, score_str = m.rsplit(":", 1)
                score = float(score_str)
                if score >= accumulate_threshold:
                    total += score
                    count += 1
            except (ValueError, AttributeError):
                pass  # corrupt member — skip gracefully

        escalated = total >= escalate_threshold
        return SessionRiskResult(
            accumulated_score=round(total, 3),
            request_count=count,
            escalated=escalated,
        )

    except Exception:
        # Redis failure is non-fatal — degrade gracefully
        return SessionRiskResult(
            accumulated_score=0.0, request_count=0, escalated=False
        )
