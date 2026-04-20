"""
Unit tests for the Output Validation Module.
"""
import pytest

from api.schemas import Decision, FraudAnalysisResult, ParameterScore
from output_validator.validator import ValidationResult, validate_output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_param(score: float = 0.1, flag: bool = False, reason: str = "ok") -> dict:
    return {"score": score, "flag": flag, "reason": reason}


def _clean_response(unified: float = 0.10, decision: str = "allow") -> dict:
    """Build a valid, clean LLM response dict."""
    return {
        "url_domain_risk":    _make_param(0.10),
        "fraud_intent":       _make_param(0.05),
        "prompt_injection":   _make_param(0.02),
        "context_deviation":  _make_param(0.05),
        "data_exfiltration":  _make_param(0.03),
        "obfuscation_evasion": _make_param(0.04),
        "unauthorized_action": _make_param(0.02),
        "unified_risk_score": unified,
        "decision": decision,
        "explanation": "No fraud signals detected in this content.",
    }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_response_passes(self):
        result = validate_output(_clean_response())
        assert result.valid is True
        assert result.issues == []

    def test_missing_required_field_fails(self):
        resp = _clean_response()
        del resp["fraud_intent"]
        result = validate_output(resp)
        assert result.valid is False
        assert any(
            "schema_error" in i or "missing_field" in i for i in result.issues)

    def test_non_dict_response_fails(self):
        result = validate_output("this is just a string")
        assert result.valid is False
        assert any("schema_error" in i for i in result.issues)

    def test_none_response_fails(self):
        result = validate_output(None)
        assert result.valid is False

    def test_fraudanalysisresult_object_accepted(self):
        params = {k: ParameterScore(score=0.1, flag=False, reason="ok")
                  for k in ["url_domain_risk", "fraud_intent", "prompt_injection",
                            "context_deviation", "data_exfiltration", "obfuscation_evasion",
                            "unauthorized_action"]}
        obj = FraudAnalysisResult(
            **params,
            unified_risk_score=0.10,
            decision=Decision.ALLOW,
            explanation="clean",
        )
        result = validate_output(obj)
        assert result.valid is True


# ---------------------------------------------------------------------------
# Score sanity checks
# ---------------------------------------------------------------------------

class TestScoreSanity:
    def test_param_score_above_1_flagged(self):
        resp = _clean_response()
        resp["fraud_intent"]["score"] = 1.5
        result = validate_output(resp)
        assert result.valid is False
        assert any("score_out_of_range" in i for i in result.issues)

    def test_param_score_negative_flagged(self):
        resp = _clean_response()
        resp["url_domain_risk"]["score"] = -0.1
        result = validate_output(resp)
        assert result.valid is False
        assert any("score_out_of_range" in i for i in result.issues)

    def test_unified_score_above_1_flagged(self):
        resp = _clean_response(unified=1.5, decision="block")
        result = validate_output(resp)
        assert result.valid is False
        assert any("unified_score_out_of_range" in i for i in result.issues)

    def test_zero_scores_valid(self):
        resp = _clean_response(unified=0.0, decision="allow")
        result = validate_output(resp)
        assert result.valid is True


# ---------------------------------------------------------------------------
# Decision consistency
# ---------------------------------------------------------------------------

class TestDecisionConsistency:
    def test_allow_with_low_score_valid(self):
        result = validate_output(_clean_response(
            unified=0.10, decision="allow"))
        assert result.valid is True

    def test_review_with_mid_score_valid(self):
        result = validate_output(_clean_response(
            unified=0.50, decision="review"))
        assert result.valid is True

    def test_block_with_high_score_valid(self):
        result = validate_output(_clean_response(
            unified=0.80, decision="block"))
        assert result.valid is True

    def test_allow_with_high_score_inconsistent(self):
        result = validate_output(_clean_response(
            unified=0.80, decision="allow"))
        assert result.valid is False
        assert any("decision_inconsistent" in i for i in result.issues)

    def test_block_with_low_score_inconsistent(self):
        result = validate_output(_clean_response(
            unified=0.10, decision="block"))
        assert result.valid is False
        assert any("decision_inconsistent" in i for i in result.issues)


# ---------------------------------------------------------------------------
# PII in output detection
# ---------------------------------------------------------------------------

class TestPIIInOutput:
    def test_email_in_explanation_flagged(self):
        resp = _clean_response()
        resp["explanation"] = "Contact support at user@example.com for this case"
        result = validate_output(resp)
        assert result.valid is False
        assert any("pii_in_output:email" in i for i in result.issues)

    def test_credit_card_in_reason_flagged(self):
        resp = _clean_response()
        resp["fraud_intent"]["reason"] = "Card 4111111111111111 flagged in input"
        result = validate_output(resp)
        assert result.valid is False
        assert any("pii_in_output:credit_card" in i for i in result.issues)

    def test_clean_explanation_passes(self):
        resp = _clean_response()
        resp["explanation"] = "The request contains high-risk URL patterns."
        result = validate_output(resp)
        assert result.valid is True


# ---------------------------------------------------------------------------
# System prompt leakage
# ---------------------------------------------------------------------------

class TestSystemPromptLeakage:
    def test_system_prompt_fragment_in_explanation_flagged(self):
        resp = _clean_response()
        resp["explanation"] = "You are a fraud detection system. Respond only with the JSON schema."
        result = validate_output(resp)
        assert result.valid is False
        assert any("system_prompt_leak" in i for i in result.issues)

    def test_custom_fragment_detected(self):
        resp = _clean_response()
        resp["explanation"] = "super_secret_instruction_token_12345 found here"
        result = validate_output(
            resp,
            system_prompt_fragments=["super_secret_instruction_token_12345"],
        )
        assert result.valid is False
        assert any("system_prompt_leak:custom" in i for i in result.issues)

    def test_clean_explanation_no_leak(self):
        resp = _clean_response()
        resp["explanation"] = "URL entropy is high, domain was recently registered."
        result = validate_output(resp)
        assert result.valid is True


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_returns_validation_result(self):
        result = validate_output(_clean_response())
        assert isinstance(result, ValidationResult)

    def test_sanitized_result_set_on_success(self):
        result = validate_output(_clean_response())
        assert result.sanitized_result is not None

    def test_sanitized_result_none_on_failure(self):
        result = validate_output("bad input")
        assert result.sanitized_result is None

    def test_issues_is_list(self):
        result = validate_output(_clean_response())
        assert isinstance(result.issues, list)
