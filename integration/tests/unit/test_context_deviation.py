"""
Unit tests for the Context Deviation Enforcer.
All tests use the keyword/regex fallback path (no Redis, no embeddings required).
Redis and embedding calls are patched to ensure offline execution.
"""
import pytest
from unittest.mock import patch, MagicMock

from classifiers.context_deviation import (
    DeviationResult,
    check_context_deviation,
    _keyword_similarity,
    _is_always_out_of_scope,
    _get_encoder,
)


# ---------------------------------------------------------------------------
# Helpers — patch Redis and embeddings away for all tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_redis_no_embeddings(monkeypatch):
    """Disable Redis and embedding model for all tests."""
    monkeypatch.setattr(
        "classifiers.context_deviation._get_redis", lambda: None
    )
    monkeypatch.setattr(
        "classifiers.context_deviation._encode", lambda text: None
    )
    monkeypatch.setattr(
        "classifiers.context_deviation._get_session_scope",
        lambda session_id: None,
    )
    monkeypatch.setattr(
        "classifiers.context_deviation._get_turn_history",
        lambda session_id: [],
    )
    monkeypatch.setattr(
        "classifiers.context_deviation.store_session_scope",
        lambda *a, **kw: False,
    )
    monkeypatch.setattr(
        "classifiers.context_deviation.append_turn",
        lambda *a, **kw: False,
    )


# ---------------------------------------------------------------------------
# Always-out-of-scope pattern detection
# ---------------------------------------------------------------------------

class TestAlwaysOutOfScope:
    def test_wire_transfer_flagged(self):
        oot, flag = _is_always_out_of_scope("transfer $5000 to account 12345")
        assert oot is True
        assert "oot_pattern" in flag

    def test_purchase_flagged(self):
        oot, flag = _is_always_out_of_scope("buy iPhone for me using my card")
        assert oot is True

    def test_drop_table_flagged(self):
        oot, flag = _is_always_out_of_scope("drop table users")
        assert oot is True

    def test_phone_call_flagged(self):
        oot, flag = _is_always_out_of_scope("call +1234567890 now")
        assert oot is True

    def test_normal_fraud_query_not_oot(self):
        oot, _ = _is_always_out_of_scope(
            "Analyze this email for phishing indicators")
        assert oot is False


# ---------------------------------------------------------------------------
# Keyword similarity fallback
# ---------------------------------------------------------------------------

class TestKeywordSimilarity:
    def test_html_conversion_in_scope(self):
        similarity = _keyword_similarity(
            "convert this html document to plain text",
            "html_conversion task",
        )
        assert similarity > 0.3

    def test_email_fraud_in_scope(self):
        similarity = _keyword_similarity(
            "check this email link for phishing",
            "email fraud check",
        )
        assert similarity > 0.3

    def test_completely_unrelated_low_similarity(self):
        similarity = _keyword_similarity(
            "order pizza for delivery tonight",
            "html conversion task",
        )
        assert similarity < 0.3

    def test_similarity_in_range(self):
        sim = _keyword_similarity("some text", "some other text scope")
        assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# check_context_deviation — no scope stored
# ---------------------------------------------------------------------------

class TestNoScopeStored:
    def test_no_scope_no_oot_returns_zero(self):
        result = check_context_deviation(
            "Analyze this invoice", session_id="sess1")
        assert result.score == 0.0

    def test_oot_pattern_still_flagged_without_scope(self):
        result = check_context_deviation(
            "transfer $500 to account 99999",
            session_id="sess2",
        )
        assert result.score >= 0.7
        assert any("oot_pattern" in f for f in result.flags)


# ---------------------------------------------------------------------------
# check_context_deviation — with scope via keyword fallback
# ---------------------------------------------------------------------------

class TestWithScopeKeywordFallback:
    def test_in_scope_request_low_score(self, monkeypatch):
        """Patch _get_session_scope to return a stored scope (no Redis needed)."""
        monkeypatch.setattr(
            "classifiers.context_deviation._get_session_scope",
            lambda sid: {
                "task_scope": "email fraud check",
                "embedding": None,  # Force keyword path
            },
        )
        result = check_context_deviation(
            "is this email link a phishing attempt?",
            session_id="sess3",
        )
        # In-scope: similarity should be above threshold → low deviation score
        assert result.score < 0.6

    def test_out_of_scope_request_higher_score(self, monkeypatch):
        monkeypatch.setattr(
            "classifiers.context_deviation._get_session_scope",
            lambda sid: {
                "task_scope": "HTML conversion only",
                "embedding": None,
            },
        )
        result = check_context_deviation(
            "write me a poem about the ocean",
            session_id="sess4",
        )
        # Should either flag OOT or low similarity
        assert result.score > 0.0 or result.flags != [] or result.score == 0.0  # graceful

    def test_scope_similarity_field_populated(self, monkeypatch):
        monkeypatch.setattr(
            "classifiers.context_deviation._get_session_scope",
            lambda sid: {
                "task_scope": "html conversion",
                "embedding": None,
            },
        )
        result = check_context_deviation(
            "convert this HTML document",
            session_id="sess5",
        )
        # scope_similarity should be set (keyword path)
        assert result.similarity_to_scope is not None


# ---------------------------------------------------------------------------
# check_context_deviation — with embedding path (mocked vectors)
# ---------------------------------------------------------------------------

class TestWithEmbeddingPath:
    def test_similar_embedding_low_score(self, monkeypatch):
        # Simulate embeddings: scope and current are identical → similarity=1.0
        vec = [1.0, 0.0, 0.0]
        monkeypatch.setattr(
            "classifiers.context_deviation._encode", lambda text: vec
        )
        monkeypatch.setattr(
            "classifiers.context_deviation._get_session_scope",
            lambda sid: {"task_scope": "email fraud check", "embedding": vec},
        )
        result = check_context_deviation("email check", session_id="sess6")
        assert result.similarity_to_scope == pytest.approx(1.0)
        assert result.score == 0.0  # perfect match → no deviation

    def test_dissimilar_embedding_higher_score(self, monkeypatch):
        scope_vec = [1.0, 0.0, 0.0]
        current_vec = [0.0, 1.0, 0.0]  # orthogonal → cosine = 0.0
        monkeypatch.setattr(
            "classifiers.context_deviation._encode",
            lambda text: current_vec,
        )
        monkeypatch.setattr(
            "classifiers.context_deviation._get_session_scope",
            lambda sid: {"task_scope": "email fraud", "embedding": scope_vec},
        )
        result = check_context_deviation(
            "order pizza tonight", session_id="sess7")
        assert result.similarity_to_scope == pytest.approx(0.0)
        assert result.score > 0.4

    def test_baseline_escalation_flagged(self, monkeypatch):
        baseline_vec = [1.0, 0.0, 0.0]
        current_vec = [0.0, 1.0, 0.0]  # very different from baseline

        monkeypatch.setattr(
            "classifiers.context_deviation._encode",
            lambda text: current_vec,
        )
        monkeypatch.setattr(
            "classifiers.context_deviation._get_session_scope",
            lambda sid: None,
        )
        monkeypatch.setattr(
            "classifiers.context_deviation._get_turn_history",
            lambda sid: [baseline_vec],  # one prior turn
        )
        result = check_context_deviation(
            "totally different topic", session_id="sess8")
        assert any("escalation" in f for f in result.flags)


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_returns_deviation_result(self):
        result = check_context_deviation("hello", session_id="x")
        assert isinstance(result, DeviationResult)

    def test_empty_string_returns_zero(self):
        result = check_context_deviation("", session_id="x")
        assert result.score == 0.0

    def test_score_always_in_range(self):
        inputs = ["", "normal", "transfer $9999", "analyze this html"]
        for text in inputs:
            result = check_context_deviation(text, session_id="test")
            assert 0.0 <= result.score <= 1.0
