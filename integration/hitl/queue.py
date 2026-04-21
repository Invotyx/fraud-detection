"""
Human-in-the-Loop (HITL) Queue
--------------------------------
Routes ambiguous (review) decisions to a PostgreSQL queue for human review.

Table schema (see alembic migration for DDL):
    hitl_queue(id, request_id, created_at, unified_risk_score,
               classifier_scores, llm_response, decision_pending,
               reviewed_by, reviewed_at, reviewer_decision, reviewer_notes,
               escalated_at)

Redis is used to track SLA deadlines (TTL = SLA seconds from enqueue).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from integration.api.schemas import Decision

# ---------------------------------------------------------------------------
# SQLAlchemy model (declarative base-free — raw table references for simplicity)
# ---------------------------------------------------------------------------

_INSERT_SQL = text("""
    INSERT INTO hitl_queue (
        id, request_id, created_at, unified_risk_score,
        classifier_scores, llm_response, decision_pending
    ) VALUES (
        :id, :request_id, :created_at, :unified_risk_score,
        :classifier_scores::jsonb, :llm_response::jsonb, :decision_pending
    )
""")

_SELECT_PENDING_SQL = text("""
    SELECT id, request_id, created_at, unified_risk_score,
           classifier_scores, decision_pending, reviewed_at,
           reviewer_decision, escalated_at
    FROM hitl_queue
    WHERE reviewed_at IS NULL
    ORDER BY created_at ASC
    LIMIT :limit
""")

_SELECT_BY_ID_SQL = text("""
    SELECT id, request_id, created_at, unified_risk_score,
           classifier_scores, llm_response, decision_pending,
           reviewed_by, reviewed_at, reviewer_decision, reviewer_notes, escalated_at
    FROM hitl_queue
    WHERE id = :id
""")

_UPDATE_DECISION_SQL = text("""
    UPDATE hitl_queue
    SET reviewed_by = :reviewed_by,
        reviewed_at = :reviewed_at,
        reviewer_decision = :reviewer_decision,
        reviewer_notes = :reviewer_notes
    WHERE id = :id
      AND reviewed_at IS NULL
    RETURNING id
""")

_ESCALATE_SQL = text("""
    UPDATE hitl_queue
    SET escalated_at = :escalated_at
    WHERE reviewed_at IS NULL
      AND escalated_at IS NULL
      AND created_at < :cutoff_time
    RETURNING id
""")

_CREATE_HITL_TABLE = text("""
    CREATE TABLE IF NOT EXISTS hitl_queue (
        id UUID PRIMARY KEY,
        request_id UUID NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        unified_risk_score DOUBLE PRECISION NOT NULL,
        classifier_scores JSONB NOT NULL,
        llm_response JSONB NULL,
        decision_pending VARCHAR(20) NULL,
        reviewed_by VARCHAR(100) NULL,
        reviewed_at TIMESTAMPTZ NULL,
        reviewer_decision VARCHAR(20) NULL,
        reviewer_notes TEXT NULL,
        escalated_at TIMESTAMPTZ NULL
    )
""")

_CREATE_HITL_REVIEWED_AT_INDEX = text(
    "CREATE INDEX IF NOT EXISTS ix_hitl_queue_reviewed_at ON hitl_queue (reviewed_at)"
)

_CREATE_HITL_CREATED_AT_INDEX = text(
    "CREATE INDEX IF NOT EXISTS ix_hitl_queue_created_at ON hitl_queue (created_at)"
)

# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

import json


async def ensure_hitl_schema(db: AsyncSession) -> None:
    """Create HITL queue tables and indexes if they do not exist."""
    await db.execute(_CREATE_HITL_TABLE)
    await db.execute(_CREATE_HITL_REVIEWED_AT_INDEX)
    await db.execute(_CREATE_HITL_CREATED_AT_INDEX)


async def enqueue(
    db: AsyncSession,
    *,
    request_id: str,
    unified_risk_score: float,
    classifier_scores: Dict[str, Any],
    llm_response: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Insert a new HITL review item.

    Returns:
        The UUID of the created queue item.
    """
    item_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    await db.execute(_INSERT_SQL, {
        "id": item_id,
        "request_id": request_id,
        "created_at": now,
        "unified_risk_score": unified_risk_score,
        "classifier_scores": json.dumps(classifier_scores),
        "llm_response": json.dumps(llm_response) if llm_response else json.dumps({}),
        "decision_pending": Decision.REVIEW.value,
    })
    await db.commit()
    return item_id


async def get_pending(
    db: AsyncSession,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return pending (unreviewed) queue items, oldest first."""
    result = await db.execute(_SELECT_PENDING_SQL, {"limit": limit})
    rows = result.mappings().all()
    return [dict(r) for r in rows]


async def get_item(
    db: AsyncSession,
    item_id: str,
) -> Optional[Dict[str, Any]]:
    """Return full detail of a single queue item by ID."""
    result = await db.execute(_SELECT_BY_ID_SQL, {"id": item_id})
    row = result.mappings().first()
    return dict(row) if row else None


async def submit_decision(
    db: AsyncSession,
    item_id: str,
    *,
    reviewer: str,
    decision: str,   # "allow" or "block"
    notes: str = "",
) -> bool:
    """
    Submit a human review decision.

    Returns:
        True if updated successfully, False if item not found or already reviewed.
    """
    if decision not in ("allow", "block"):
        raise ValueError(f"Invalid reviewer decision: {decision!r}. Must be 'allow' or 'block'.")

    now = datetime.now(timezone.utc)
    result = await db.execute(_UPDATE_DECISION_SQL, {
        "id": item_id,
        "reviewed_by": reviewer,
        "reviewed_at": now,
        "reviewer_decision": decision,
        "reviewer_notes": notes,
    })
    await db.commit()
    return result.rowcount > 0


async def escalate_stale(
    db: AsyncSession,
    sla_seconds: int = 3600,
) -> List[str]:
    """
    Mark unreviewed items exceeding the SLA as escalated.

    Returns:
        List of escalated item IDs.
    """
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=sla_seconds)
    result = await db.execute(_ESCALATE_SQL, {
        "escalated_at": now,
        "cutoff_time": cutoff,
    })
    await db.commit()
    rows = result.fetchall()
    return [str(r[0]) for r in rows]
