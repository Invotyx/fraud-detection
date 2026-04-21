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


# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

_SESSION_KEY_PREFIX = "session_risk:"
WINDOW_SECONDS = 3600          # rolling 1-hour window
ACCUMULATE_THRESHOLD = 0.40    # minimum single-request score that counts
SESSION_ESCALATE_THRESHOLD = 1.20  # sum that triggers session-level escalation


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
        pipe.zremrangebyscore(key, 0, now - WINDOW_SECONDS)
        # 3. Refresh TTL so idle sessions expire automatically
        pipe.expire(key, int(WINDOW_SECONDS))
        # 4. Fetch all remaining members
        pipe.zrange(key, 0, -1)
        _, _, _, members = await pipe.execute()

        total = 0.0
        count = 0
        for m in members:
            try:
                _, score_str = m.rsplit(":", 1)
                score = float(score_str)
                if score >= ACCUMULATE_THRESHOLD:
                    total += score
                    count += 1
            except (ValueError, AttributeError):
                pass  # corrupt member — skip gracefully

        escalated = total >= SESSION_ESCALATE_THRESHOLD
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
