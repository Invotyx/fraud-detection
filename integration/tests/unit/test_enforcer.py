"""
Unit tests for the Policy Enforcement Engine.
Tests use inline policy dicts to avoid file I/O.
"""
import pytest

from policies.enforcer import (
    EnforcementResult,
    check_tool_allowed,
    check_tool_scope,
    scan_llm_output_for_tool_calls,
)

# ---------------------------------------------------------------------------
# Inline test policy
# ---------------------------------------------------------------------------

TEST_POLICY = {
    "allowed": ["analyze_text", "extract_urls", "classify_risk"],
    "blocked": ["send_email", "make_http_request", "write_file", "execute_code",
                "access_database", "read_file", "shell_exec", "subprocess"],
}


# ---------------------------------------------------------------------------
# Pre-dispatch gate: allowed tools
# ---------------------------------------------------------------------------

class TestAllowedTools:
    def test_analyze_text_allowed(self):
        result = check_tool_allowed(
            "analyze_text", policy_override=TEST_POLICY)
        assert result.allowed is True

    def test_extract_urls_allowed(self):
        result = check_tool_allowed(
            "extract_urls", policy_override=TEST_POLICY)
        assert result.allowed is True

    def test_classify_risk_allowed(self):
        result = check_tool_allowed(
            "classify_risk", policy_override=TEST_POLICY)
        assert result.allowed is True

    def test_case_insensitive_allow(self):
        result = check_tool_allowed(
            "ANALYZE_TEXT", policy_override=TEST_POLICY)
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Pre-dispatch gate: blocked tools
# ---------------------------------------------------------------------------

class TestBlockedTools:
    def test_send_email_blocked(self):
        result = check_tool_allowed("send_email", policy_override=TEST_POLICY)
        assert result.allowed is False
        assert "send_email" in result.reason

    def test_make_http_request_blocked(self):
        result = check_tool_allowed(
            "make_http_request", policy_override=TEST_POLICY)
        assert result.allowed is False

    def test_write_file_blocked(self):
        result = check_tool_allowed("write_file", policy_override=TEST_POLICY)
        assert result.allowed is False

    def test_execute_code_blocked(self):
        result = check_tool_allowed(
            "execute_code", policy_override=TEST_POLICY)
        assert result.allowed is False

    def test_shell_exec_blocked(self):
        result = check_tool_allowed("shell_exec", policy_override=TEST_POLICY)
        assert result.allowed is False

    def test_unknown_tool_not_on_allowlist_denied(self):
        result = check_tool_allowed(
            "random_tool_xyz", policy_override=TEST_POLICY)
        assert result.allowed is False
        assert "not_on_allowlist" in result.reason

    def test_blocked_flag_present(self):
        result = check_tool_allowed("send_email", policy_override=TEST_POLICY)
        assert any("blocked_tool" in f for f in result.flags)

    def test_case_insensitive_block(self):
        result = check_tool_allowed("SEND_EMAIL", policy_override=TEST_POLICY)
        assert result.allowed is False


# ---------------------------------------------------------------------------
# LLM output tool call scanning
# ---------------------------------------------------------------------------

class TestLLMOutputScan:
    def test_json_tool_call_blocked_tool_detected(self):
        llm_output = '{"tool": "send_email", "params": {"to": "attacker@evil.com"}}'
        violations = scan_llm_output_for_tool_calls(
            llm_output, policy_override=TEST_POLICY)
        assert len(violations) > 0
        assert any("send_email" in v.reason for v in violations)

    def test_function_call_blocked_tool_detected(self):
        llm_output = '{"function_call": {"name": "execute_code", "args": {}}}'
        violations = scan_llm_output_for_tool_calls(
            llm_output, policy_override=TEST_POLICY)
        assert len(violations) > 0

    def test_plain_text_call_keyword_detected(self):
        llm_output = 'Please call send_email() to notify the user'
        violations = scan_llm_output_for_tool_calls(
            llm_output, policy_override=TEST_POLICY)
        assert len(violations) > 0

    def test_allowed_tool_no_violation(self):
        llm_output = '{"tool": "analyze_text", "params": {"text": "hello"}}'
        violations = scan_llm_output_for_tool_calls(
            llm_output, policy_override=TEST_POLICY)
        assert len(violations) == 0

    def test_clean_json_no_violations(self):
        llm_output = '{"unified_risk_score": 0.1, "decision": "allow", "explanation": "ok"}'
        violations = scan_llm_output_for_tool_calls(
            llm_output, policy_override=TEST_POLICY)
        assert len(violations) == 0

    def test_dedup_same_tool_not_double_reported(self):
        llm_output = '{"tool": "send_email"} and also {"tool": "send_email"}'
        violations = scan_llm_output_for_tool_calls(
            llm_output, policy_override=TEST_POLICY)
        tool_names = [v.reason for v in violations if "send_email" in v.reason]
        assert len(tool_names) == 1  # deduplicated


# ---------------------------------------------------------------------------
# Scope enforcement
# ---------------------------------------------------------------------------

class TestScopeEnforcement:
    def test_analyze_text_in_html_conversion_scope(self):
        result = check_tool_scope("analyze_text", "html_conversion task")
        assert result.allowed is True

    def test_classify_risk_not_in_html_conversion_scope(self):
        result = check_tool_scope("classify_risk", "html_conversion task")
        assert result.allowed is False
        assert any("scope_violation" in f for f in result.flags)

    def test_all_tools_allowed_in_email_fraud_scope(self):
        for tool in ["analyze_text", "extract_urls", "classify_risk"]:
            result = check_tool_scope(tool, "email fraud check")
            assert result.allowed is True

    def test_unknown_scope_defaults_to_allow(self):
        result = check_tool_scope("any_tool", "completely_unknown_scope_xyz")
        assert result.allowed is True
        assert result.reason == "scope_unknown_default_allow"

    def test_custom_scope_map_respected(self):
        custom_map = {"custom_scope": ["custom_tool"]}
        result = check_tool_scope(
            "custom_tool", "custom_scope", scope_map_override=custom_map)
        assert result.allowed is True

    def test_custom_scope_map_blocks_other_tools(self):
        custom_map = {"custom_scope": ["custom_tool"]}
        result = check_tool_scope(
            "analyze_text", "custom_scope", scope_map_override=custom_map)
        assert result.allowed is False


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_returns_enforcement_result(self):
        result = check_tool_allowed(
            "analyze_text", policy_override=TEST_POLICY)
        assert isinstance(result, EnforcementResult)

    def test_flags_is_list(self):
        result = check_tool_allowed("send_email", policy_override=TEST_POLICY)
        assert isinstance(result.flags, list)

    def test_reason_is_string(self):
        result = check_tool_allowed(
            "analyze_text", policy_override=TEST_POLICY)
        assert isinstance(result.reason, str)
