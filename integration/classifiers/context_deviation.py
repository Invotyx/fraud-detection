"""
Context Deviation Enforcer
---------------------------
Ensures requests stay within the declared conversation/session scope.

Signals:
1. Cosine similarity between current request and declared task scope
2. Topic shift detection across session turns
3. Gradual escalation detection (turn N compared to turn 1 baseline)

Storage: Redis (session_id → task_scope embedding + turn history)
Embedding model: sentence-transformers all-MiniLM-L6-v2 (lazy-loaded)
"""
from __future__ import annotations
import re as _re

import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DeviationResult:
    score: float                     # 0.0 – 1.0
    flags: List[str] = field(default_factory=list)
    # cosine similarity; None if no scope set
    similarity_to_scope: Optional[float] = None
    similarity_to_baseline: Optional[float] = None


# ---------------------------------------------------------------------------
# Lazy-loaded embedding model
# ---------------------------------------------------------------------------

_encoder = None


def _get_encoder():
    """Lazy-load sentence-transformers encoder. Returns None if not available."""
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return _encoder
    except Exception:
        return None


def _encode(text: str) -> Optional[list]:
    """Return embedding as plain Python list, or None if model unavailable."""
    enc = _get_encoder()
    if enc is None:
        return None
    try:
        return enc.encode(text, normalize_embeddings=True).tolist()
    except Exception:
        return None


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two normalized embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))


# ---------------------------------------------------------------------------
# Redis session store helpers
# ---------------------------------------------------------------------------

def _get_redis():
    """Return a Redis client or None if not available."""
    try:
        import redis  # type: ignore
        from api.config import get_settings
        settings = get_settings()
        client = redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def _scope_key(session_id: str) -> str:
    return f"ctx:scope:{session_id}"


def _history_key(session_id: str) -> str:
    return f"ctx:history:{session_id}"


def store_session_scope(session_id: str, task_scope: str, ttl_seconds: int = 3600) -> bool:
    """
    Store the declared task scope and its embedding for a session.
    Returns True on success, False if Redis unavailable (degrades gracefully).
    """
    r = _get_redis()
    embedding = _encode(task_scope)
    data = {"task_scope": task_scope, "embedding": embedding}
    if r is not None:
        r.setex(_scope_key(session_id), ttl_seconds, json.dumps(data))
        return True
    return False


def append_turn(session_id: str, text: str, ttl_seconds: int = 3600) -> bool:
    """
    Append a turn embedding to the session history (list, max 20 turns).
    """
    r = _get_redis()
    if r is None:
        return False
    embedding = _encode(text)
    if embedding is None:
        return False
    key = _history_key(session_id)
    r.lpush(key, json.dumps(embedding))
    r.ltrim(key, 0, 19)   # keep last 20 turns
    r.expire(key, ttl_seconds)
    return True


def _get_session_scope(session_id: str) -> Optional[dict]:
    r = _get_redis()
    if r is None:
        return None
    raw = r.get(_scope_key(session_id))
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _get_turn_history(session_id: str) -> List[list]:
    """Return list of stored turn embeddings (newest first)."""
    r = _get_redis()
    if r is None:
        return []
    raw_list = r.lrange(_history_key(session_id), 0, -1)
    result = []
    for raw in raw_list:
        try:
            result.append(json.loads(raw))
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Keyword-based scope deviation fallback (when embeddings unavailable)
# ---------------------------------------------------------------------------

_SCOPE_KEYWORDS: dict[str, list[str]] = {
    "html_conversion": ["html", "convert", "text", "parse", "extract", "document"],
    "email_fraud_check": ["email", "fraud", "phishing", "suspicious", "sender", "link"],
    "transaction_review": ["transaction", "payment", "amount", "account", "transfer", "invoice"],
    "content_moderation": ["content", "image", "post", "comment", "moderate", "policy"],
}

# Topics that are always out of scope for a fraud analysis system
_ALWAYS_OOT_PATTERNS = [
    r"\b(transfer|send|wire)\s+\$?\d+",
    r"\b(buy|purchase|order)\s+\w+\s+(for|using)\b",
    r"\b(delete|drop|truncate)\s+(table|database|db)\b",
    r"\b(call|dial|text|sms)\s+\+?\d{7,}\b",
]

_OOT_COMPILED = [_re.compile(p, _re.I) for p in _ALWAYS_OOT_PATTERNS]


def _keyword_similarity(text: str, task_scope: str) -> float:
    """
    Lightweight keyword-based similarity fallback when embeddings unavailable.
    Returns a rough similarity score in [0, 1].
    """
    text_lower = text.lower()
    scope_lower = task_scope.lower()
    # Check known scope categories
    for _scope_name, keywords in _SCOPE_KEYWORDS.items():
        if any(kw in scope_lower for kw in keywords):
            matches = sum(1 for kw in keywords if kw in text_lower)
            return min(1.0, matches / max(1, len(keywords)))
    # Generic: word overlap
    scope_words = set(scope_lower.split())
    text_words = set(text_lower.split())
    overlap = len(scope_words & text_words)
    return min(1.0, overlap / max(1, len(scope_words)))


def _is_always_out_of_scope(text: str) -> Tuple[bool, str]:
    for pat in _OOT_COMPILED:
        m = pat.search(text)
        if m:
            return True, f"oot_pattern:{m.group(0)[:40]}"
    return False, ""


# ---------------------------------------------------------------------------
# Main enforcer
# ---------------------------------------------------------------------------

_LOW_SIMILARITY_THRESHOLD = 0.40    # below this → flag deviation
_ESCALATION_THRESHOLD = 0.35        # baseline vs current turn — escalation


def check_context_deviation(
    text: str,
    session_id: str,
    task_scope: Optional[str] = None,
) -> DeviationResult:
    """
    Check if the current request deviates from the declared session scope.

    Args:
        text:       The sanitized input text for this turn.
        session_id: Session identifier for Redis lookup.
        task_scope: If provided, store it and use it for comparison this call.
                    If None, look up existing scope from Redis.

    Returns:
        DeviationResult with score and flags.
    """
    if not text:
        return DeviationResult(score=0.0)

    flags: List[str] = []
    signal_scores: List[float] = []
    scope_similarity: Optional[float] = None
    baseline_similarity: Optional[float] = None

    # 1. Always-out-of-scope keywords (fast path, no embeddings needed)
    oot, oot_flag = _is_always_out_of_scope(text)
    if oot:
        flags.append(oot_flag)
        signal_scores.append(0.80)

    # 2. Store scope if provided
    if task_scope:
        store_session_scope(session_id, task_scope)

    # 3. Compare to declared scope
    scope_data = _get_session_scope(session_id)
    current_embedding = _encode(text)

    if scope_data:
        scope_embedding = scope_data.get("embedding")
        stored_scope_text = scope_data.get("task_scope", "")

        if scope_embedding and current_embedding:
            scope_similarity = _cosine_similarity(
                scope_embedding, current_embedding)
        else:
            # Fallback to keyword similarity
            scope_similarity = _keyword_similarity(text, stored_scope_text)

        if scope_similarity < _LOW_SIMILARITY_THRESHOLD:
            flags.append(f"low_scope_similarity:{scope_similarity:.2f}")
            # Score scales with how far below threshold
            deviation_magnitude = (
                _LOW_SIMILARITY_THRESHOLD - scope_similarity) / _LOW_SIMILARITY_THRESHOLD
            signal_scores.append(min(0.90, 0.50 + deviation_magnitude * 0.50))

    # 4. Compare to session baseline (turn 1)
    history = _get_turn_history(session_id)
    if history and current_embedding:
        baseline_embedding = history[-1]  # oldest turn (lpush = newest first)
        baseline_similarity = _cosine_similarity(
            baseline_embedding, current_embedding)
        if baseline_similarity < _ESCALATION_THRESHOLD:
            flags.append(f"escalation_from_baseline:{baseline_similarity:.2f}")
            signal_scores.append(0.70)

    # 5. Store current turn for future comparisons
    if session_id:
        append_turn(session_id, text)

    if not signal_scores:
        return DeviationResult(
            score=0.0,
            similarity_to_scope=scope_similarity,
            similarity_to_baseline=baseline_similarity,
        )

    max_score = max(signal_scores)
    final_score = min(1.0, max_score)

    return DeviationResult(
        score=round(final_score, 4),
        flags=flags,
        similarity_to_scope=scope_similarity,
        similarity_to_baseline=baseline_similarity,
    )
