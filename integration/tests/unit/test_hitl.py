"""
Unit tests for the HITL Queue.
Uses an in-memory SQLite database (aiosqlite) to avoid needing a real PostgreSQL server.
"""
from sqlalchemy import text as t
import json
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------------------
# In-memory SQLite setup (mirrors the hitl_queue schema for testing)
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

CREATE_HITL_TABLE = """
CREATE TABLE IF NOT EXISTS hitl_queue (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    unified_risk_score REAL NOT NULL,
    classifier_scores TEXT NOT NULL,
    llm_response TEXT,
    decision_pending TEXT,
    reviewed_by TEXT,
    reviewed_at TEXT,
    reviewer_decision TEXT,
    reviewer_notes TEXT,
    escalated_at TEXT
)
"""


@pytest_asyncio.fixture
async def db_session():
    """Provide an async SQLite session for testing."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(text(CREATE_HITL_TABLE))

    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

    await engine.dispose()


# ---------------------------------------------------------------------------
# SQLite-compatible queue operations (SQLite doesn't support ::jsonb cast)
# ---------------------------------------------------------------------------


_INSERT = t("""
    INSERT INTO hitl_queue (
        id, request_id, created_at, unified_risk_score,
        classifier_scores, llm_response, decision_pending
    ) VALUES (
        :id, :request_id, :created_at, :unified_risk_score,
        :classifier_scores, :llm_response, :decision_pending
    )
""")

_SELECT_PENDING = t("""
    SELECT id, request_id, created_at, unified_risk_score,
           decision_pending, reviewed_at
    FROM hitl_queue
    WHERE reviewed_at IS NULL
    ORDER BY created_at ASC
    LIMIT :limit
""")

_SELECT_BY_ID = t("""
    SELECT * FROM hitl_queue WHERE id = :id
""")

_UPDATE_DECISION = t("""
    UPDATE hitl_queue
    SET reviewed_by = :reviewed_by,
        reviewed_at = :reviewed_at,
        reviewer_decision = :reviewer_decision,
        reviewer_notes = :reviewer_notes
    WHERE id = :id AND reviewed_at IS NULL
""")

_ESCALATE = t("""
    UPDATE hitl_queue
    SET escalated_at = :escalated_at
    WHERE reviewed_at IS NULL
      AND escalated_at IS NULL
      AND created_at < :cutoff_time
""")


async def _enqueue(db, *, request_id, unified_risk_score, classifier_scores):
    item_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(_INSERT, {
        "id": item_id,
        "request_id": request_id,
        "created_at": now,
        "unified_risk_score": unified_risk_score,
        "classifier_scores": json.dumps(classifier_scores),
        "llm_response": "{}",
        "decision_pending": "review",
    })
    await db.commit()
    return item_id


async def _get_pending(db, limit=50):
    result = await db.execute(_SELECT_PENDING, {"limit": limit})
    return result.mappings().all()


async def _get_item(db, item_id):
    result = await db.execute(_SELECT_BY_ID, {"id": item_id})
    row = result.mappings().first()
    return dict(row) if row else None


async def _submit_decision(db, item_id, *, reviewer, decision, notes=""):
    now = datetime.now(timezone.utc).isoformat()
    result = await db.execute(_UPDATE_DECISION, {
        "id": item_id,
        "reviewed_by": reviewer,
        "reviewed_at": now,
        "reviewer_decision": decision,
        "reviewer_notes": notes,
    })
    await db.commit()
    return result.rowcount > 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEnqueue:
    async def test_enqueue_creates_item(self, db_session):
        item_id = await _enqueue(
            db_session,
            request_id=str(uuid.uuid4()),
            unified_risk_score=0.55,
            classifier_scores={"fraud_intent": 0.55},
        )
        assert item_id is not None
        item = await _get_item(db_session, item_id)
        assert item is not None
        assert item["unified_risk_score"] == pytest.approx(0.55)

    async def test_enqueue_decision_pending_is_review(self, db_session):
        item_id = await _enqueue(
            db_session,
            request_id=str(uuid.uuid4()),
            unified_risk_score=0.50,
            classifier_scores={},
        )
        item = await _get_item(db_session, item_id)
        assert item["decision_pending"] == "review"

    async def test_enqueue_reviewed_at_is_null(self, db_session):
        item_id = await _enqueue(
            db_session,
            request_id=str(uuid.uuid4()),
            unified_risk_score=0.50,
            classifier_scores={},
        )
        item = await _get_item(db_session, item_id)
        assert item["reviewed_at"] is None


@pytest.mark.asyncio
class TestGetPending:
    async def test_pending_returns_unreviewed_items(self, db_session):
        for i in range(3):
            await _enqueue(
                db_session,
                request_id=str(uuid.uuid4()),
                unified_risk_score=0.5 + i * 0.05,
                classifier_scores={},
            )
        pending = await _get_pending(db_session)
        assert len(pending) == 3

    async def test_reviewed_items_excluded(self, db_session):
        item_id = await _enqueue(
            db_session,
            request_id=str(uuid.uuid4()),
            unified_risk_score=0.55,
            classifier_scores={},
        )
        await _submit_decision(db_session, item_id, reviewer="analyst", decision="allow")
        pending = await _get_pending(db_session)
        assert all(p["id"] != item_id for p in pending)

    async def test_limit_respected(self, db_session):
        for _ in range(10):
            await _enqueue(
                db_session,
                request_id=str(uuid.uuid4()),
                unified_risk_score=0.50,
                classifier_scores={},
            )
        pending = await _get_pending(db_session, limit=5)
        assert len(pending) <= 5


@pytest.mark.asyncio
class TestSubmitDecision:
    async def test_allow_decision_recorded(self, db_session):
        item_id = await _enqueue(
            db_session,
            request_id=str(uuid.uuid4()),
            unified_risk_score=0.55,
            classifier_scores={},
        )
        success = await _submit_decision(
            db_session, item_id, reviewer="analyst1", decision="allow", notes="Looks safe"
        )
        assert success is True
        item = await _get_item(db_session, item_id)
        assert item["reviewer_decision"] == "allow"
        assert item["reviewed_by"] == "analyst1"
        assert item["reviewed_at"] is not None

    async def test_block_decision_recorded(self, db_session):
        item_id = await _enqueue(
            db_session,
            request_id=str(uuid.uuid4()),
            unified_risk_score=0.65,
            classifier_scores={},
        )
        await _submit_decision(db_session, item_id, reviewer="analyst2", decision="block")
        item = await _get_item(db_session, item_id)
        assert item["reviewer_decision"] == "block"

    async def test_already_reviewed_not_updated_twice(self, db_session):
        item_id = await _enqueue(
            db_session,
            request_id=str(uuid.uuid4()),
            unified_risk_score=0.60,
            classifier_scores={},
        )
        await _submit_decision(db_session, item_id, reviewer="a1", decision="allow")
        # Second attempt should affect 0 rows
        result = await _submit_decision(db_session, item_id, reviewer="a2", decision="block")
        assert result is False

    async def test_nonexistent_id_returns_false(self, db_session):
        result = await _submit_decision(
            db_session, str(uuid.uuid4()), reviewer="a1", decision="allow"
        )
        assert result is False


@pytest.mark.asyncio
class TestGetItem:
    async def test_get_item_returns_dict(self, db_session):
        item_id = await _enqueue(
            db_session,
            request_id=str(uuid.uuid4()),
            unified_risk_score=0.52,
            classifier_scores={"fraud_intent": 0.52},
        )
        item = await _get_item(db_session, item_id)
        assert isinstance(item, dict)
        assert "id" in item

    async def test_nonexistent_item_returns_none(self, db_session):
        item = await _get_item(db_session, str(uuid.uuid4()))
        assert item is None
