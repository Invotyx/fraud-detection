"""
Vector Store — pgvector CRUD operations.

All functions take an asyncpg Connection (or compatible object).
The pgvector codec is registered once per connection via init_vector_tables().
"""
from __future__ import annotations

import datetime
import hashlib
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_embedding_cfg() -> dict:
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "classifiers.yaml"
    )
    with open(cfg_path) as fh:
        return yaml.safe_load(fh).get("embeddings", {})


def _get_dim() -> int:
    return int(_load_embedding_cfg().get("dim", 768))


def _get_session_ttl() -> int:
    return int(_load_embedding_cfg().get("session_ttl_seconds", 3600))


def _get_turn_history_max() -> int:
    return int(_load_embedding_cfg().get("turn_history_max", 20))


# ---------------------------------------------------------------------------
# Codec helper
# ---------------------------------------------------------------------------

async def ensure_vector_codec(conn: Any) -> None:
    """
    Register the pgvector asyncpg codec on *conn* if not already registered.

    Safe to call multiple times (asyncpg overwrites with the same codec).
    Call this once per raw asyncpg connection before any vector I/O to handle
    connections that were created after the initial pool warm-up.
    """
    try:
        from pgvector.asyncpg import register_vector  # type: ignore
        await register_vector(conn)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

async def init_vector_tables(conn: Any) -> None:
    """
    Register the pgvector type codec on *conn* and create tables if they do
    not exist.  Safe to call multiple times (idempotent).
    """
    try:
        from pgvector.asyncpg import register_vector  # type: ignore
        await register_vector(conn)
    except Exception:
        # pgvector Python package not installed — ops will degrade gracefully
        pass

    dim = _get_dim()

    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS session_embeddings (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id   TEXT NOT NULL,
            turn_n       INTEGER NOT NULL DEFAULT 0,
            content_hash TEXT,
            embedding    vector({dim}) NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at   TIMESTAMPTZ NOT NULL
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_session_embeddings_session_turn
        ON session_embeddings (session_id, turn_n)
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_session_embeddings_expires
        ON session_embeddings (expires_at)
    """)
    await conn.execute(f"""
        CREATE TABLE IF NOT EXISTS fraud_patterns (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            pattern_text TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            embedding    vector({dim}) NOT NULL,
            severity     FLOAT NOT NULL DEFAULT 0.8,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS ix_fraud_patterns_type
        ON fraud_patterns (pattern_type)
    """)


# ---------------------------------------------------------------------------
# Session scope embedding (turn_n = 0)
# ---------------------------------------------------------------------------

async def upsert_scope_embedding(
    conn: Any,
    session_id: str,
    text: str,
    embedding: List[float],
    ttl_seconds: Optional[int] = None,
) -> None:
    """Store (or replace) the declared task-scope embedding for a session."""
    if ttl_seconds is None:
        ttl_seconds = _get_session_ttl()
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=ttl_seconds
    )
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    # Replace existing scope row for this session
    await conn.execute(
        "DELETE FROM session_embeddings WHERE session_id = $1 AND turn_n = 0",
        session_id,
    )
    await conn.execute(
        """
        INSERT INTO session_embeddings
            (session_id, turn_n, content_hash, embedding, expires_at)
        VALUES ($1, 0, $2, $3, $4)
        """,
        session_id,
        content_hash,
        np.asarray(embedding, dtype=np.float32).tolist(),
        expires_at,
    )


async def get_scope_embedding(
    conn: Any,
    session_id: str,
) -> Optional[Tuple[List[float], str]]:
    """
    Return (embedding_list, content_hash) for the session scope, or None if
    not found or expired.
    """
    row = await conn.fetchrow(
        """
        SELECT embedding, content_hash
        FROM session_embeddings
        WHERE session_id = $1 AND turn_n = 0 AND expires_at > now()
        """,
        session_id,
    )
    if row is None:
        return None
    emb = row["embedding"]
    if isinstance(emb, str):
        emb = [float(x) for x in emb.strip("[]").split(",")]
    return list(emb), (row["content_hash"] or "")


# ---------------------------------------------------------------------------
# Turn history embeddings (turn_n > 0)
# ---------------------------------------------------------------------------

async def append_turn_embedding(
    conn: Any,
    session_id: str,
    text: str,
    embedding: List[float],
    ttl_seconds: Optional[int] = None,
    max_turns: Optional[int] = None,
) -> None:
    """Append a request-turn embedding. Prunes history beyond *max_turns*."""
    if ttl_seconds is None:
        ttl_seconds = _get_session_ttl()
    if max_turns is None:
        max_turns = _get_turn_history_max()

    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=ttl_seconds
    )
    content_hash = hashlib.sha256(text.encode()).hexdigest()

    # Determine next turn_n
    row = await conn.fetchrow(
        """
        SELECT COALESCE(MAX(turn_n), 0) AS max_turn
        FROM session_embeddings
        WHERE session_id = $1 AND turn_n > 0
        """,
        session_id,
    )
    next_turn = (row["max_turn"] or 0) + 1

    await conn.execute(
        """
        INSERT INTO session_embeddings
            (session_id, turn_n, content_hash, embedding, expires_at)
        VALUES ($1, $2, $3, $4, $5)
        """,
        session_id,
        next_turn,
        content_hash,
        np.asarray(embedding, dtype=np.float32).tolist(),
        expires_at,
    )

    # Prune turns exceeding the max window (keep newest max_turns)
    if next_turn > max_turns:
        await conn.execute(
            """
            DELETE FROM session_embeddings
            WHERE session_id = $1 AND turn_n > 0
              AND turn_n NOT IN (
                  SELECT turn_n FROM session_embeddings
                  WHERE session_id = $1 AND turn_n > 0
                  ORDER BY turn_n DESC
                  LIMIT $2
              )
            """,
            session_id,
            max_turns,
        )


async def get_turn_history(
    conn: Any,
    session_id: str,
    limit: Optional[int] = None,
) -> List[List[float]]:
    """Return the most-recent turn embeddings for the session (newest first)."""
    if limit is None:
        limit = _get_turn_history_max()
    rows = await conn.fetch(
        """
        SELECT embedding
        FROM session_embeddings
        WHERE session_id = $1 AND turn_n > 0 AND expires_at > now()
        ORDER BY turn_n DESC
        LIMIT $2
        """,
        session_id,
        limit,
    )
    result = []
    for row in rows:
        emb = row["embedding"]
        if isinstance(emb, str):
            emb = [float(x) for x in emb.strip("[]").split(",")]
        result.append(list(emb))
    return result


# ---------------------------------------------------------------------------
# Fraud pattern search (RAG)
# ---------------------------------------------------------------------------

async def search_fraud_patterns(
    conn: Any,
    embedding: List[float],
    top_k: int,
    min_similarity: float,
) -> List[dict]:
    """
    Find *top_k* fraud patterns whose cosine similarity to *embedding* is at
    least *min_similarity*.

    Returns list of dicts: {text, type, severity, similarity}.
    """
    # pgvector: embedding <=> query = cosine distance = 1 − cosine_similarity
    rows = await conn.fetch(
        """
        SELECT pattern_text, pattern_type, severity,
               1.0 - (embedding <=> $1) AS similarity
        FROM fraud_patterns
        WHERE 1.0 - (embedding <=> $1) >= $3
        ORDER BY embedding <=> $1
        LIMIT $2
        """,
        np.asarray(embedding, dtype=np.float32).tolist(),
        top_k,
        min_similarity,
    )
    return [
        {
            "text": row["pattern_text"],
            "type": row["pattern_type"],
            "severity": float(row["severity"]),
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Cross-session drift detection
# ---------------------------------------------------------------------------

async def find_similar_recent_sessions(
    conn: Any,
    embedding: List[float],
    current_session_id: str,
    top_k: int,
    min_similarity: float,
    lookback_seconds: int,
) -> List[dict]:
    """
    Find recent scope embeddings (from OTHER sessions) that are similar to
    *embedding*.  Used for cross-session drift / coordinated attack detection.

    Returns list of dicts: {session_id, similarity, created_at}.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=lookback_seconds
    )
    rows = await conn.fetch(
        """
        SELECT session_id,
               1.0 - (embedding <=> $1) AS similarity,
               created_at
        FROM session_embeddings
        WHERE turn_n = 0
          AND session_id != $2
          AND created_at >= $5
          AND 1.0 - (embedding <=> $1) >= $4
        ORDER BY embedding <=> $1
        LIMIT $3
        """,
        np.asarray(embedding, dtype=np.float32).tolist(),
        current_session_id,
        top_k,
        min_similarity,
        cutoff,
    )
    return [
        {
            "session_id": row["session_id"],
            "similarity": float(row["similarity"]),
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

async def purge_expired_session_embeddings(conn: Any) -> int:
    """Delete expired session embedding rows. Returns count removed."""
    result = await conn.execute(
        "DELETE FROM session_embeddings WHERE expires_at <= now()"
    )
    # asyncpg returns "DELETE N" string
    try:
        return int(result.split()[-1])
    except (IndexError, ValueError):
        return 0
