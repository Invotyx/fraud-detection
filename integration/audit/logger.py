"""
Audit Logger — Phase 12
Append-only structured audit trail for every pipeline request.
No UPDATE/DELETE methods are exposed; enforcement is also at DB level.
All sensitive fields are stored as SHA-256 hashes only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

import structlog
from pythonjsonlogger import jsonlogger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

# ---------------------------------------------------------------------------
# Structured logger configuration
# ---------------------------------------------------------------------------

_json_handler = logging.StreamHandler()
_json_handler.setFormatter(
    jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
)

logging.basicConfig(handlers=[_json_handler], level=logging.INFO)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

_log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(value: str) -> str:
    """Return the hex SHA-256 hash of a UTF-8 string."""
    return hashlib.sha256(value.encode("utf-8")).digest().hex()


def _safe_json(obj: Any) -> str:
    """Serialise obj to a compact JSON string for JSONB columns."""
    if obj is None:
        return "null"
    return json.dumps(obj, default=str)


# ---------------------------------------------------------------------------
# SQL (no ORM — append-only, no UPDATE/DELETE)
# ---------------------------------------------------------------------------

_INSERT_AUDIT = text(
    """
    INSERT INTO audit_log (
        id,
        trace_id,
        input_hash,
        raw_input_hash,
        classifier_scores,
        llm_response,
        unified_risk_score,
        decision,
        flags,
        hitl_required,
        processing_time_ms,
        created_at
    ) VALUES (
        :id,
        :trace_id,
        :input_hash,
        :raw_input_hash,
        :classifier_scores,
        :llm_response,
        :unified_risk_score,
        :decision,
        :flags,
        :hitl_required,
        :processing_time_ms,
        NOW()
    )
    """
)

_SELECT_BY_TRACE = text(
    """
    SELECT id, trace_id, created_at, input_hash, raw_input_hash,
           classifier_scores, llm_response, unified_risk_score,
           decision, flags, hitl_required, processing_time_ms
    FROM audit_log
    WHERE trace_id = :trace_id
    ORDER BY created_at DESC
    LIMIT 1
    """
)

_CREATE_AUDIT_TABLE = text(
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id UUID PRIMARY KEY,
        trace_id UUID NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        input_hash VARCHAR(64) NOT NULL,
        raw_input_hash VARCHAR(64) NOT NULL,
        classifier_scores JSONB NOT NULL,
        llm_response JSONB NULL,
        unified_risk_score DOUBLE PRECISION NOT NULL,
        decision VARCHAR(20) NOT NULL,
        flags JSONB NOT NULL,
        hitl_required BOOLEAN NOT NULL DEFAULT FALSE,
        processing_time_ms INTEGER NULL
    )
    """
)

_CREATE_AUDIT_TRACE_INDEX = text(
    "CREATE INDEX IF NOT EXISTS ix_audit_log_trace_id ON audit_log (trace_id)"
)

_CREATE_AUDIT_CREATED_AT_INDEX = text(
    "CREATE INDEX IF NOT EXISTS ix_audit_log_created_at ON audit_log (created_at)"
)


# ---------------------------------------------------------------------------
# Public API — append-only
# ---------------------------------------------------------------------------

async def log_request(
    db: AsyncConnection,
    *,
    trace_id: UUID,
    sanitized_text: str,
    raw_text: str,
    classifier_scores: Dict[str, Any],
    llm_response: Optional[Dict[str, Any]],
    unified_risk_score: float,
    decision: str,
    flags: Dict[str, Any],
    hitl_required: bool = False,
    processing_time_ms: int = 0,
) -> UUID:
    """
    Insert one immutable audit record.

    Sensitive content (``sanitized_text``, ``raw_text``) is **never stored
    in plain text** — only their SHA-256 digests are persisted.

    Returns the newly created audit record UUID.
    """
    record_id = uuid4()

    # Hash sensitive content — only hashes stored in DB
    input_hash = _sha256(sanitized_text)
    raw_input_hash = _sha256(raw_text)

    params: Dict[str, Any] = {
        "id": str(record_id),
        "trace_id": str(trace_id),
        "input_hash": input_hash,
        "raw_input_hash": raw_input_hash,
        "classifier_scores": _safe_json(classifier_scores),
        "llm_response": _safe_json(llm_response),
        "unified_risk_score": float(unified_risk_score),
        "decision": str(decision),
        "flags": _safe_json(flags),
        "hitl_required": bool(hitl_required),
        "processing_time_ms": int(processing_time_ms),
    }

    await db.execute(_INSERT_AUDIT, params)

    _log.info(
        "audit_record_written",
        audit_id=str(record_id),
        trace_id=str(trace_id),
        decision=decision,
        unified_risk_score=unified_risk_score,
        hitl_required=hitl_required,
        processing_time_ms=processing_time_ms,
        input_hash=input_hash,
        # raw_input_hash intentionally omitted from structured log
    )

    return record_id


async def ensure_audit_schema(db: AsyncConnection) -> None:
    """Create audit tables and indexes if they do not exist."""
    await db.execute(_CREATE_AUDIT_TABLE)
    await db.execute(_CREATE_AUDIT_TRACE_INDEX)
    await db.execute(_CREATE_AUDIT_CREATED_AT_INDEX)


async def get_by_trace(db: AsyncConnection, trace_id: UUID) -> Optional[Dict[str, Any]]:
    """
    Retrieve the most recent audit entry for the given trace_id.
    Read-only helper — no mutations exposed.
    """
    row = (await db.execute(_SELECT_BY_TRACE, {"trace_id": str(trace_id)})).fetchone()
    if row is None:
        return None
    return dict(row._mapping)


# ---------------------------------------------------------------------------
# Stage-level structured logging (no DB — in-memory structured events)
# ---------------------------------------------------------------------------

def log_stage(stage: str, trace_id: UUID, **kwargs: Any) -> None:
    """
    Emit a structured log event for a single pipeline stage.
    Called by each pipeline step; does NOT write to the database.
    Sensitive values must be pre-hashed before passing as kwargs.
    """
    _log.info(
        "pipeline_stage",
        stage=stage,
        trace_id=str(trace_id),
        **kwargs,
    )
