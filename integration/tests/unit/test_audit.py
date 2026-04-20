"""
Unit tests for audit/logger.py — Phase 12

Uses SQLite in-memory (aiosqlite) to avoid requiring PostgreSQL.
"""
from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from integration.audit.logger import _sha256, get_by_trace, log_request, log_stage

# ---------------------------------------------------------------------------
# Schema — mirrors Alembic migration 0001
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              TEXT PRIMARY KEY,
    trace_id        TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    input_hash      TEXT NOT NULL,
    raw_input_hash  TEXT NOT NULL,
    classifier_scores TEXT NOT NULL,
    llm_response    TEXT,
    unified_risk_score REAL NOT NULL,
    decision        TEXT NOT NULL,
    flags           TEXT NOT NULL,
    hitl_required   INTEGER NOT NULL DEFAULT 0,
    processing_time_ms INTEGER
);
"""


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.execute(text(_DDL))
    async with engine.connect() as conn:
        async with conn.begin():
            yield conn
    await engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _insert(db: AsyncConnection, **overrides) -> UUID:
    defaults = dict(
        trace_id=uuid4(),
        sanitized_text="hello world",
        raw_text="<b>hello world</b>",
        classifier_scores={"url_domain_risk": 0.1, "fraud_intent": 0.0},
        llm_response={"decision": "allow"},
        unified_risk_score=0.1,
        decision="allow",
        flags={},
        hitl_required=False,
        processing_time_ms=42,
    )
    defaults.update(overrides)
    return await log_request(db, **defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLogCreation:
    @pytest.mark.asyncio
    async def test_returns_uuid(self, db: AsyncConnection):
        record_id = await _insert(db)
        assert isinstance(record_id, UUID)

    @pytest.mark.asyncio
    async def test_row_exists_after_insert(self, db: AsyncConnection):
        trace_id = uuid4()
        await _insert(db, trace_id=trace_id)
        row = await get_by_trace(db, trace_id)
        assert row is not None

    @pytest.mark.asyncio
    async def test_trace_id_preserved(self, db: AsyncConnection):
        trace_id = uuid4()
        await _insert(db, trace_id=trace_id)
        row = await get_by_trace(db, trace_id)
        assert row["trace_id"] == str(trace_id)

    @pytest.mark.asyncio
    async def test_decision_preserved(self, db: AsyncConnection):
        trace_id = uuid4()
        await _insert(db, trace_id=trace_id, decision="block")
        row = await get_by_trace(db, trace_id)
        assert row["decision"] == "block"

    @pytest.mark.asyncio
    async def test_unified_risk_score_preserved(self, db: AsyncConnection):
        trace_id = uuid4()
        await _insert(db, trace_id=trace_id, unified_risk_score=0.75)
        row = await get_by_trace(db, trace_id)
        assert abs(row["unified_risk_score"] - 0.75) < 1e-9

    @pytest.mark.asyncio
    async def test_hitl_required_preserved(self, db: AsyncConnection):
        trace_id = uuid4()
        await _insert(db, trace_id=trace_id, hitl_required=True)
        row = await get_by_trace(db, trace_id)
        # SQLite stores bools as 0/1
        assert bool(row["hitl_required"])

    @pytest.mark.asyncio
    async def test_processing_time_preserved(self, db: AsyncConnection):
        trace_id = uuid4()
        await _insert(db, trace_id=trace_id, processing_time_ms=123)
        row = await get_by_trace(db, trace_id)
        assert row["processing_time_ms"] == 123


class TestHashStorage:
    @pytest.mark.asyncio
    async def test_raw_text_not_in_db(self, db: AsyncConnection):
        """Raw user input must never appear in plain text in any auditable column."""
        trace_id = uuid4()
        sensitive_text = "super-secret-input-that-should-not-be-stored"
        await _insert(db, trace_id=trace_id, raw_text=sensitive_text)
        row = await get_by_trace(db, trace_id)

        # Check all string columns
        for col in ("input_hash", "raw_input_hash", "classifier_scores",
                    "llm_response", "flags"):
            val = row.get(col) or ""
            assert sensitive_text not in str(val), (
                f"Column '{col}' must not contain plain raw text"
            )

    @pytest.mark.asyncio
    async def test_sanitized_text_not_in_db(self, db: AsyncConnection):
        trace_id = uuid4()
        sanitized = "visible-sanitized-content-no-store"
        await _insert(db, trace_id=trace_id, sanitized_text=sanitized)
        row = await get_by_trace(db, trace_id)
        for col in ("input_hash", "raw_input_hash", "classifier_scores",
                    "llm_response", "flags"):
            val = row.get(col) or ""
            assert sanitized not in str(val)

    @pytest.mark.asyncio
    async def test_input_hash_is_sha256_of_sanitized(self, db: AsyncConnection):
        trace_id = uuid4()
        sanitized = "predictable sanitized text"
        await _insert(db, trace_id=trace_id, sanitized_text=sanitized)
        row = await get_by_trace(db, trace_id)
        assert row["input_hash"] == _sha256(sanitized)

    @pytest.mark.asyncio
    async def test_raw_input_hash_is_sha256_of_raw(self, db: AsyncConnection):
        trace_id = uuid4()
        raw = "<script>raw content</script>"
        await _insert(db, trace_id=trace_id, raw_text=raw)
        row = await get_by_trace(db, trace_id)
        assert row["raw_input_hash"] == _sha256(raw)

    @pytest.mark.asyncio
    async def test_hash_length_64_chars(self, db: AsyncConnection):
        trace_id = uuid4()
        await _insert(db, trace_id=trace_id)
        row = await get_by_trace(db, trace_id)
        assert len(row["input_hash"]) == 64
        assert len(row["raw_input_hash"]) == 64

    @pytest.mark.asyncio
    async def test_different_inputs_produce_different_hashes(self, db: AsyncConnection):
        t1, t2 = uuid4(), uuid4()
        await _insert(db, trace_id=t1, sanitized_text="aaa", raw_text="aaa")
        await _insert(db, trace_id=t2, sanitized_text="bbb", raw_text="bbb")
        r1 = await get_by_trace(db, t1)
        r2 = await get_by_trace(db, t2)
        assert r1["input_hash"] != r2["input_hash"]


class TestGetByTrace:
    @pytest.mark.asyncio
    async def test_missing_trace_returns_none(self, db: AsyncConnection):
        result = await get_by_trace(db, uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_classifier_scores_stored_as_json(self, db: AsyncConnection):
        trace_id = uuid4()
        scores = {"url_domain_risk": 0.9, "fraud_intent": 0.8}
        await _insert(db, trace_id=trace_id, classifier_scores=scores)
        row = await get_by_trace(db, trace_id)
        stored = json.loads(row["classifier_scores"])
        assert stored["url_domain_risk"] == pytest.approx(0.9)
        assert stored["fraud_intent"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_flags_stored_as_json(self, db: AsyncConnection):
        trace_id = uuid4()
        flags = {"blocklist_match": True, "injection_rule": False}
        await _insert(db, trace_id=trace_id, flags=flags)
        row = await get_by_trace(db, trace_id)
        stored = json.loads(row["flags"])
        assert stored["blocklist_match"] is True

    @pytest.mark.asyncio
    async def test_none_llm_response_stored_as_null(self, db: AsyncConnection):
        trace_id = uuid4()
        await _insert(db, trace_id=trace_id, llm_response=None)
        row = await get_by_trace(db, trace_id)
        # "null" JSON string or None both acceptable
        assert row["llm_response"] in (None, "null")


class TestHashHelper:
    def test_sha256_returns_hex_string(self):
        result = _sha256("test")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_sha256_deterministic(self):
        assert _sha256("hello") == _sha256("hello")

    def test_sha256_collision_resistance(self):
        assert _sha256("a") != _sha256("b")


class TestLogStage:
    def test_log_stage_does_not_raise(self):
        """log_stage emits structured logs; must not raise."""
        log_stage("sanitize", trace_id=uuid4(), removed=3, took_ms=1)
        log_stage("classify", trace_id=uuid4(), score=0.5)
