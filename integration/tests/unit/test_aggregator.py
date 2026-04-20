"""
Unit tests for the Risk Scoring Aggregation Engine.
All tests pass weights and thresholds directly (no file I/O).
"""
import pytest

from api.schemas import Decision
from risk_engine.aggregator import (
    AggregationResult,
    HardOverrideFlags,
    aggregate,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

WEIGHTS = {
    "url_domain_risk": 0.15,
    "fraud_intent": 0.20,
    "prompt_injection": 0.20,
    "context_deviation": 0.10,
    "data_exfiltration": 0.15,
    "obfuscation_evasion": 0.10,
    "unauthorized_action": 0.10,
}

THRESHOLDS = {
    "allow": 0.30,
    "review": 0.70,
    "hard_block_single_param": 0.90,
}

ALL_ZERO = {k: 0.0 for k in WEIGHTS}
ALL_HIGH = {k: 0.95 for k in WEIGHTS}
MID_SCORES = {k: 0.50 for k in WEIGHTS}


# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------

class TestDecisionThresholds:
    def test_all_zero_scores_allow(self):
        result = aggregate(ALL_ZERO, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.decision == Decision.ALLOW
        assert result.unified_risk_score < 0.30

    def test_all_mid_scores_review(self):
        result = aggregate(MID_SCORES, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.decision == Decision.REVIEW
        assert 0.30 <= result.unified_risk_score < 0.70

    def test_all_high_scores_block(self):
        # All 0.95 → single-param hard override kicks in first
        result = aggregate(ALL_HIGH, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.decision == Decision.BLOCK

    def test_low_score_boundary(self):
        # Exactly at allow threshold boundary
        scores = {k: 0.29 for k in WEIGHTS}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.decision == Decision.ALLOW

    def test_at_review_threshold(self):
        # Weighted sum = 0.30 → review
        scores = {k: 0.30 for k in WEIGHTS}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.decision == Decision.REVIEW

    def test_at_block_threshold(self):
        scores = {k: 0.70 for k in WEIGHTS}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.decision == Decision.BLOCK


# ---------------------------------------------------------------------------
# Hard overrides
# ---------------------------------------------------------------------------

class TestHardOverrides:
    def test_url_blocklist_match_blocks_regardless(self):
        flags = HardOverrideFlags(url_blocklist_match=True)
        result = aggregate(ALL_ZERO, hard_flags=flags,
                           weights_override=WEIGHTS, thresholds_override=THRESHOLDS)
        assert result.decision == Decision.BLOCK
        assert result.hard_override is True
        assert result.override_reason == "url_blocklist_match"
        assert result.unified_risk_score == 1.0

    def test_injection_rule_match_blocks_regardless(self):
        flags = HardOverrideFlags(injection_rule_match=True)
        result = aggregate(ALL_ZERO, hard_flags=flags,
                           weights_override=WEIGHTS, thresholds_override=THRESHOLDS)
        assert result.decision == Decision.BLOCK
        assert result.hard_override is True
        assert result.override_reason == "injection_rule_match"

    def test_single_param_above_09_hard_blocks(self):
        scores = {**ALL_ZERO, "prompt_injection": 0.95}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.decision == Decision.BLOCK
        assert result.hard_override is True
        assert "prompt_injection" in result.override_reason

    def test_single_param_exactly_09_no_hard_block(self):
        # Exactly 0.90 is NOT above the threshold (> not >=)
        scores = {**ALL_ZERO, "fraud_intent": 0.90}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.hard_override is False

    def test_url_blocklist_overrides_all_zero_scores(self):
        flags = HardOverrideFlags(url_blocklist_match=True)
        result = aggregate(ALL_ZERO, hard_flags=flags,
                           weights_override=WEIGHTS, thresholds_override=THRESHOLDS)
        assert result.decision == Decision.BLOCK


# ---------------------------------------------------------------------------
# Weighted calculation correctness
# ---------------------------------------------------------------------------

class TestWeightedCalculation:
    def test_known_weighted_sum(self):
        # Only fraud_intent=1.0 (weight 0.20) → unified = 0.20 / 1.0 = 0.20
        scores = {**ALL_ZERO, "fraud_intent": 1.0}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.unified_risk_score == pytest.approx(0.20, abs=1e-4)
        assert result.decision == Decision.ALLOW

    def test_two_params_sum(self):
        # fraud_intent=1.0 (0.20) + url_domain_risk=1.0 (0.15) = 0.35
        scores = {**ALL_ZERO, "fraud_intent": 1.0, "url_domain_risk": 1.0}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.unified_risk_score == pytest.approx(0.35, abs=1e-4)
        assert result.decision == Decision.REVIEW

    def test_all_ones_unified_score_is_1(self):
        scores = {k: 1.0 for k in WEIGHTS}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        # Hard override kicks in, but even without it score would be 1.0
        assert result.unified_risk_score == 1.0

    def test_unified_score_clamped_to_1(self):
        scores = {k: 1.5 for k in WEIGHTS}  # Above 1.0 inputs
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.unified_risk_score <= 1.0

    def test_unified_score_never_negative(self):
        scores = {k: -0.5 for k in WEIGHTS}  # Below 0 inputs
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.unified_risk_score >= 0.0


# ---------------------------------------------------------------------------
# Missing and partial parameter scores
# ---------------------------------------------------------------------------

class TestPartialScores:
    def test_missing_params_default_to_zero(self):
        # Only provide 3 of 7 parameters
        scores = {"fraud_intent": 0.5, "prompt_injection": 0.5}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        # (0.5*0.20 + 0.5*0.20) / 1.0 = 0.20 → allow
        assert result.decision == Decision.ALLOW

    def test_empty_scores_returns_allow(self):
        result = aggregate({}, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.decision == Decision.ALLOW
        assert result.unified_risk_score == 0.0


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_returns_aggregation_result(self):
        result = aggregate(ALL_ZERO, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert isinstance(result, AggregationResult)

    def test_parameter_scores_preserved(self):
        scores = {**ALL_ZERO, "fraud_intent": 0.6}
        result = aggregate(scores, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.parameter_scores["fraud_intent"] == 0.6

    def test_no_hard_flags_no_override(self):
        result = aggregate(ALL_ZERO, weights_override=WEIGHTS,
                           thresholds_override=THRESHOLDS)
        assert result.hard_override is False
        assert result.override_reason == ""
