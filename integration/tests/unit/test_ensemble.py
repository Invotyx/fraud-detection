"""
Unit tests for the Fraud Model Ensemble.
"""
import asyncio
import pytest
import pytest_asyncio

from classifiers.ensemble import (
    DEFAULT_MODELS,
    EnsembleModel,
    EnsembleResult,
    ModelScore,
    _keyword_fraud_scorer,
    _rule_based_url_scorer,
    run_ensemble,
)


# ---------------------------------------------------------------------------
# Sync scorer unit tests
# ---------------------------------------------------------------------------

class TestRuleURLScorer:
    def test_uses_url_risk_score_feature(self):
        score = _rule_based_url_scorer({"url_risk_score": 0.75})
        assert score == pytest.approx(0.75)

    def test_zero_when_missing(self):
        score = _rule_based_url_scorer({})
        assert score == 0.0

    def test_clamped_to_1(self):
        score = _rule_based_url_scorer({"url_risk_score": 2.0})
        assert score <= 1.0


class TestKeywordFraudScorer:
    def test_fraud_keywords_increase_score(self):
        score = _keyword_fraud_scorer(
            {"content": "urgent: verify your account now"})
        assert score > 0.0

    def test_clean_content_low_score(self):
        score = _keyword_fraud_scorer(
            {"content": "Invoice for consulting services rendered"})
        assert score < 0.3

    def test_multiple_keywords_higher_score(self):
        multi = _keyword_fraud_scorer({
            "content": "urgent immediate action verify your account click here now"
        })
        single = _keyword_fraud_scorer({"content": "urgent"})
        assert multi >= single

    def test_empty_content_zero(self):
        score = _keyword_fraud_scorer({"content": ""})
        assert score == 0.0

    def test_missing_content_zero(self):
        score = _keyword_fraud_scorer({})
        assert score == 0.0


# ---------------------------------------------------------------------------
# Async ensemble runner
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunEnsemble:
    async def test_returns_ensemble_result(self):
        result = await run_ensemble({"content": "hello"})
        assert isinstance(result, EnsembleResult)

    async def test_scores_list_length_matches_models(self):
        result = await run_ensemble({"content": "test"}, models=DEFAULT_MODELS)
        assert len(result.scores) == len(DEFAULT_MODELS)

    async def test_combined_score_in_range(self):
        result = await run_ensemble({"content": "test"})
        assert 0.0 <= result.combined_score <= 1.0

    async def test_placeholder_model_marked_not_implemented(self):
        result = await run_ensemble({"content": "test"})
        xgb = next((s for s in result.scores if s.model_name ==
                   "xgboost_transaction"), None)
        assert xgb is not None
        assert xgb.not_implemented is True

    async def test_placeholder_excluded_from_combined_score(self):
        # With only the placeholder model, score should be 0.0 (no available models)
        placeholder_only = [
            EnsembleModel(name="placeholder", weight=1.0,
                          scorer=lambda f: (_ for _ in ()).throw(NotImplementedError()))
        ]
        result = await run_ensemble({"content": "test"}, models=placeholder_only)
        assert result.combined_score == 0.0
        assert "placeholder" in result.unavailable_models

    async def test_real_model_contributes_to_score(self):
        models = [
            EnsembleModel(name="fixed", weight=1.0, scorer=lambda f: 0.80),
        ]
        result = await run_ensemble({}, models=models)
        assert result.combined_score == pytest.approx(0.80)

    async def test_error_model_goes_to_unavailable(self):
        def bad_scorer(f):
            raise ValueError("exploded")

        models = [
            EnsembleModel(name="broken", weight=1.0, scorer=bad_scorer),
        ]
        result = await run_ensemble({}, models=models)
        assert "broken" in result.unavailable_models
        assert result.combined_score == 0.0

    async def test_timeout_model_goes_to_unavailable(self):
        import time

        def slow_scorer(f):
            time.sleep(10)  # will timeout
            return 0.5

        models = [
            EnsembleModel(name="slow_model", weight=1.0,
                          scorer=slow_scorer, timeout_seconds=0.05),
        ]
        result = await run_ensemble({}, models=models)
        slow = next(
            (s for s in result.scores if s.model_name == "slow_model"), None)
        assert slow is not None
        assert slow.error == "timeout"
        assert "slow_model" in result.unavailable_models

    async def test_mixed_models_weighted_correctly(self):
        models = [
            EnsembleModel(name="m1", weight=0.6, scorer=lambda f: 1.0),
            EnsembleModel(name="m2", weight=0.4, scorer=lambda f: 0.0),
        ]
        result = await run_ensemble({}, models=models)
        # (1.0*0.6 + 0.0*0.4) / 1.0 = 0.6
        assert result.combined_score == pytest.approx(0.60, abs=1e-4)

    async def test_fraud_content_raises_score(self):
        result = await run_ensemble({
            "content": "urgent: verify your account click here now limited time",
            "url_risk_score": 0.0,
        }, models=[
            EnsembleModel(name="kw", weight=1.0, scorer=_keyword_fraud_scorer),
        ])
        assert result.combined_score > 0.2

    async def test_latency_tracked(self):
        result = await run_ensemble({"content": "test"})
        for score in result.scores:
            assert score.latency_ms >= 0.0
