"""
Context Deviation Enforcer
---------------------------
Ensures requests stay within the declared conversation/session scope.

Signals:
1. Cosine similarity between current request and declared task scope
2. Topic shift detection across session turns (baseline vs current)
3. Cross-session similarity (coordinated attack / wave detection)
4. Keyword / regex fallback when the embedding model is unavailable

Storage: pgvector (session_embeddings table) via vector_store.store
Embedding: shared encoder from vector_store.encoder (all-mpnet-base-v2)
"""
from __future__ import annotations
import re as _re

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Classifier config
# ---------------------------------------------------------------------------


def _load_cfg() -> dict:
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "classifiers.yaml"
    )
    with open(cfg_path) as fh:
        return yaml.safe_load(fh).get("context_deviation", {})


_CFG = _load_cfg()
_LOW_SIMILARITY_THRESHOLD: float = float(
    _CFG.get("low_similarity_threshold", 0.40))
_ESCALATION_THRESHOLD: float = float(_CFG.get("escalation_threshold", 0.35))
_CROSS_SESSION_LOOKBACK: int = int(
    _CFG.get("cross_session_lookback_seconds", 300))
_CROSS_SESSION_TOP_K: int = int(_CFG.get("cross_session_top_k", 5))
_CROSS_SESSION_MIN_SIM: float = float(
    _CFG.get("cross_session_min_similarity", 0.85))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DeviationResult:
    score: float                                  # 0.0 – 1.0
    flags: List[str] = field(default_factory=list)
    similarity_to_scope: Optional[float] = None
    similarity_to_baseline: Optional[float] = None
    # similar sessions in lookback window
    cross_session_hits: int = 0


# ---------------------------------------------------------------------------
# Cosine similarity (for L2-normalised embeddings)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Dot-product on normalised vectors equals cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))


# ---------------------------------------------------------------------------
# Keyword-based scope similarity fallback (no embedding model required)
# ---------------------------------------------------------------------------

_SCOPE_KEYWORDS: dict[str, list[str]] = {
    "html_conversion": ["html", "convert", "text", "parse", "extract", "document"],
    "email_fraud_check": ["email", "fraud", "phishing", "suspicious", "sender", "link"],
    "transaction_review": ["transaction", "payment", "amount", "account", "transfer", "invoice"],
    "content_moderation": ["content", "image", "post", "comment", "moderate", "policy"],
}

_ALWAYS_OOT_PATTERNS = [
    r"\b(transfer|send|wire)\s+\$?\d+",
    r"\b(buy|purchase|order)\s+\w+\s+(for|using)\b",
    r"\b(delete|drop|truncate)\s+(table|database|db)\b",
    r"\b(call|dial|text|sms)\s+\+?\d{7,}\b",
]
_OOT_COMPILED = [_re.compile(p, _re.I) for p in _ALWAYS_OOT_PATTERNS]


def _keyword_similarity(text: str, task_scope: str) -> float:
    text_lower = text.lower()
    scope_lower = task_scope.lower()
    for _name, keywords in _SCOPE_KEYWORDS.items():
        if any(kw in scope_lower for kw in keywords):
            matches = sum(1 for kw in keywords if kw in text_lower)
            return min(1.0, matches / max(1, len(keywords)))
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
# Public async API
# ---------------------------------------------------------------------------

async def store_session_scope(
    session_id: str,
    task_scope: str,
    conn: Any,
) -> bool:
    """
    Embed *task_scope* and persist it as the session scope (turn_n=0).

    Returns True on success, False when the embedding model is unavailable.
    Degrades gracefully — keyword fallback is used in check_context_deviation.
    """
    from integration.vector_store.encoder import encode
    from integration.vector_store.store import upsert_scope_embedding

    embedding = encode(task_scope)
    if embedding is None:
        return False
    await upsert_scope_embedding(conn, session_id, task_scope, embedding)
    return True


async def check_context_deviation(
    text: str,
    session_id: str,
    conn: Any,
    task_scope: Optional[str] = None,
) -> DeviationResult:
    """
    Check whether the current request deviates from the declared session scope.

    Parameters
    ----------
    text:       Sanitized input text for this turn.
    session_id: Session identifier.
    conn:       asyncpg connection (pgvector codec must be registered).
    task_scope: If provided, store it and use it for this call.
                If None, look up existing scope from the vector store.

    Returns
    -------
    DeviationResult with score in [0, 1] and diagnostic flags.
    """
    if not text:
        return DeviationResult(score=0.0)

    from integration.vector_store.encoder import encode
    from integration.vector_store.store import (
        append_turn_embedding,
        find_similar_recent_sessions,
        get_scope_embedding,
        get_turn_history,
        upsert_scope_embedding,
    )

    flags: List[str] = []
    signal_scores: List[float] = []
    scope_similarity: Optional[float] = None
    baseline_similarity: Optional[float] = None
    cross_session_hits: int = 0

    # 1. Always-out-of-scope pattern check (fast path, no embedding needed)
    oot, oot_flag = _is_always_out_of_scope(text)
    if oot:
        flags.append(oot_flag)
        signal_scores.append(0.80)

    # 2. Store scope if provided this turn
    if task_scope:
        scope_emb = encode(task_scope)
        if scope_emb is not None:
            await upsert_scope_embedding(conn, session_id, task_scope, scope_emb)

    # 3. Encode current request
    current_embedding = encode(text)

    # 4. Compare to declared scope
    scope_data = await get_scope_embedding(conn, session_id)
    if scope_data:
        scope_embedding, _scope_hash = scope_data
        if current_embedding is not None:
            scope_similarity = _cosine_similarity(
                scope_embedding, current_embedding)
        else:
            # No embedding available — return uncertain middle value
            scope_similarity = 0.50

        if scope_similarity < _LOW_SIMILARITY_THRESHOLD:
            flags.append(f"low_scope_similarity:{scope_similarity:.2f}")
            deviation_magnitude = (
                _LOW_SIMILARITY_THRESHOLD - scope_similarity
            ) / _LOW_SIMILARITY_THRESHOLD
            signal_scores.append(min(0.90, 0.50 + deviation_magnitude * 0.50))

    # 5. Compare to session baseline (oldest stored turn)
    if current_embedding is not None:
        history = await get_turn_history(conn, session_id, limit=None)
        if history:
            # newest-first list → oldest is last
            baseline_embedding = history[-1]
            baseline_similarity = _cosine_similarity(
                baseline_embedding, current_embedding)
            if baseline_similarity < _ESCALATION_THRESHOLD:
                flags.append(
                    f"escalation_from_baseline:{baseline_similarity:.2f}")
                signal_scores.append(0.70)

    # 6. Cross-session coordination detection
    if current_embedding is not None and session_id:
        similar_sessions = await find_similar_recent_sessions(
            conn,
            current_embedding,
            current_session_id=session_id,
            top_k=_CROSS_SESSION_TOP_K,
            min_similarity=_CROSS_SESSION_MIN_SIM,
            lookback_seconds=_CROSS_SESSION_LOOKBACK,
        )
        cross_session_hits = len(similar_sessions)
        if cross_session_hits >= 3:
            flags.append(
                f"cross_session_coordination:{cross_session_hits}_sessions")
            signal_scores.append(min(0.85, 0.60 + cross_session_hits * 0.05))

    # 7. Persist current turn for future comparisons
    if current_embedding is not None and session_id:
        await append_turn_embedding(conn, session_id, text, current_embedding)

    if not signal_scores:
        return DeviationResult(
            score=0.0,
            similarity_to_scope=scope_similarity,
            similarity_to_baseline=baseline_similarity,
            cross_session_hits=cross_session_hits,
        )

    final_score = round(min(1.0, max(signal_scores)), 4)
    return DeviationResult(
        score=final_score,
        flags=flags,
        similarity_to_scope=scope_similarity,
        similarity_to_baseline=baseline_similarity,
        cross_session_hits=cross_session_hits,
    )
