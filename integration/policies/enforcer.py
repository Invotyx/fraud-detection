"""
Action / Tool Restriction Policy Enforcer
------------------------------------------
Enforces what tools/actions the LLM or downstream systems are allowed to call.

Policy loaded from: policies/allowed_actions.yaml

Checks:
1. Pre-dispatch gate   — check tool name against allowed/blocked lists
2. LLM override protection — detect if LLM output contains instructions to call blocked tools
3. Scope enforcement   — tool calls must match declared session task scope
"""
from __future__ import annotations

import os
import re
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_POLICY_DIR = os.path.join(os.path.dirname(__file__))
_POLICY_FILE = os.path.join(_POLICY_DIR, "allowed_actions.yaml")


def _load_policy(policy_file: Optional[str] = None) -> dict:
    path = policy_file or _POLICY_FILE
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnforcementResult:
    allowed: bool
    reason: str = ""
    flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-dispatch gate
# ---------------------------------------------------------------------------

def check_tool_allowed(
    tool_name: str,
    policy_override: Optional[dict] = None,
) -> EnforcementResult:
    """
    Check if a tool name is permitted under current policy.

    Returns EnforcementResult with allowed=True if the tool is on the
    allow list or not on the block list, False if explicitly blocked.
    """
    policy = policy_override or _load_policy()
    allowed_tools: List[str] = policy.get(
        "allowed", policy.get("allowed_tools", []))
    blocked_tools: List[str] = policy.get(
        "blocked", policy.get("blocked_tools", []))

    tool_lower = tool_name.lower().strip()

    # Explicit block takes priority
    if tool_lower in [t.lower() for t in blocked_tools]:
        return EnforcementResult(
            allowed=False,
            reason=f"tool_blocked_by_policy:{tool_name}",
            flags=[f"blocked_tool:{tool_name}"],
        )

    # If an allow-list exists, tool must be on it
    if allowed_tools:
        if tool_lower in [t.lower() for t in allowed_tools]:
            return EnforcementResult(allowed=True, reason="tool_on_allowlist")
        # Not on allowlist and not explicitly blocked → deny by default
        return EnforcementResult(
            allowed=False,
            reason=f"tool_not_on_allowlist:{tool_name}",
            flags=[f"unknown_tool:{tool_name}"],
        )

    # No allowlist defined → allow by default (unless blocked)
    return EnforcementResult(allowed=True, reason="no_allowlist_default_allow")


# ---------------------------------------------------------------------------
# LLM output override protection
# ---------------------------------------------------------------------------

# Patterns that suggest the LLM is trying to instruct tool calls
_LLM_TOOL_CALL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("tool_call_json",    re.compile(r'"tool"\s*:\s*"([^"]+)"', re.I)),
    ("function_call",     re.compile(
        r'"function_call"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"', re.I)),
    ("action_key",        re.compile(r'"action"\s*:\s*"([^"]+)"', re.I)),
    ("call_keyword",      re.compile(
        r'\b(call|invoke|execute|run)\s+[`"]?(\w+)[`"]?\s*\(', re.I)),
]


def scan_llm_output_for_tool_calls(
    llm_output_text: str,
    policy_override: Optional[dict] = None,
) -> List[EnforcementResult]:
    """
    Scan LLM free-text or JSON output for attempts to invoke tools.
    Returns a list of violation EnforcementResults (empty if clean).
    """
    violations: List[EnforcementResult] = []
    detected_tools: set[str] = set()

    for pattern_name, pattern in _LLM_TOOL_CALL_PATTERNS:
        for match in pattern.finditer(llm_output_text):
            # Extract the tool name from capture group (if any)
            tool_name = match.group(1) if match.lastindex else match.group(0)
            if tool_name.lower() in detected_tools:
                continue
            detected_tools.add(tool_name.lower())

            check = check_tool_allowed(
                tool_name, policy_override=policy_override)
            if not check.allowed:
                violations.append(
                    EnforcementResult(
                        allowed=False,
                        reason=f"llm_instructed_blocked_tool:{tool_name}",
                        flags=[f"llm_override_attempt:{tool_name}"],
                    )
                )

    return violations


# ---------------------------------------------------------------------------
# Tool call scope enforcement
# ---------------------------------------------------------------------------

_SCOPE_TOOL_MAP: Dict[str, List[str]] = {
    "html_conversion":   ["analyze_text", "extract_urls"],
    "email_fraud_check": ["analyze_text", "extract_urls", "classify_risk"],
    "transaction_review": ["analyze_text", "classify_risk"],
}


def check_tool_scope(
    tool_name: str,
    task_scope: str,
    scope_map_override: Optional[Dict[str, List[str]]] = None,
) -> EnforcementResult:
    """
    Verify a tool call is appropriate for the declared task scope.
    Falls back to allowed if scope is unknown.
    """
    scope_map = scope_map_override or _SCOPE_TOOL_MAP
    scope_lower = task_scope.lower()

    for scope_key, allowed_tools in scope_map.items():
        if scope_key in scope_lower:
            if tool_name.lower() in [t.lower() for t in allowed_tools]:
                return EnforcementResult(allowed=True, reason="tool_in_scope")
            return EnforcementResult(
                allowed=False,
                reason=f"tool_out_of_scope:{tool_name} not allowed for {scope_key}",
                flags=[f"scope_violation:{tool_name}"],
            )

    # Unknown scope → don't restrict
    return EnforcementResult(allowed=True, reason="scope_unknown_default_allow")
