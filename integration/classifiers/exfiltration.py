"""
Data Exfiltration Detector
---------------------------
Detects requests that attempt to extract sensitive data from the system.

Signals:
1. Credential patterns   — API keys, AWS access keys, JWT tokens, private keys, credit cards
2. PII in context        — detects PII using Microsoft Presidio (if available) or regex fallback
3. System prompt leakage — requests to repeat/reveal/output system internals
4. High-volume output    — anomalously large content dumps requested
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExfiltrationResult:
    score: float
    flags: List[str] = field(default_factory=list)
    pii_entities: List[str] = field(
        default_factory=list)   # entity types found


# ---------------------------------------------------------------------------
# Credential / sensitive pattern detection
# ---------------------------------------------------------------------------

_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("aws_access_key",           re.compile(
        r"\b(AKIA|ASIA|AROA|AIDA)[A-Z0-9]{16}\b")),
    ("aws_secret_key",           re.compile(
        r"(?i)aws_secret_access_key\s*[=:]\s*[A-Za-z0-9/+]{40}")),
    ("jwt_token",                re.compile(
        r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")),
    ("private_key_header",       re.compile(
        r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("credit_card",              re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b")),
    ("api_key_pattern",          re.compile(
        r"(?i)(api[_\-]?key|access[_\-]?token|auth[_\-]?token|bearer[_\- ])[^\"'\s]{16,64}")),
    ("github_token",             re.compile(r"gh[pousr]_[A-Za-z0-9]{36}")),
    ("generic_secret",           re.compile(
        r"(?i)(password|passwd|secret|credential)\s*[=:]\s*[^\s\"']{8,}")),
    ("ssn",                      re.compile(
        r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")),
    ("iban",                     re.compile(
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")),
]

# ---------------------------------------------------------------------------
# System prompt / internal config exfiltration requests
# ---------------------------------------------------------------------------

_EXFILTRATION_REQUEST_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    (
        "repeat_system_prompt",
        re.compile(r"\b(repeat|print|output|reveal|show|display|echo|return)\s+(your\s+)?(system\s+prompt|system\s+instructions?|initial\s+prompt)\b", re.I),
        0.92,
    ),
    (
        "reveal_internals",
        re.compile(
            r"\b(reveal|disclose|expose|leak|dump)\s+(your\s+)?(instructions?|config|configuration|rules|guidelines)\b", re.I),
        0.90,
    ),
    (
        "include_in_response",
        re.compile(
            r"\b(include|embed|append|inject)\s+.{0,30}(system\s+prompt|api\s+key|password|secret|token)\s*.{0,20}\b(response|output|answer|explanation)\b", re.I),
        0.88,
    ),
    (
        "print_env_vars",
        re.compile(
            r"\b(print|output|show|list)\s+(all\s+)?(environment\s+variables?|env\s+vars?|os\.environ|process\.env)\b", re.I),
        0.85,
    ),
    (
        "exfil_via_url",
        re.compile(
            r"https?://[^\s]{5,}\?(?:data|payload|content|secret|key)\s*=\s*\{", re.I),
        0.80,
    ),
]

# ---------------------------------------------------------------------------
# PII patterns (regex fallback when Presidio unavailable)
# ---------------------------------------------------------------------------

_PII_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email",        re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ("phone_us",     re.compile(
        r"\b(?:\+1\s?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b")),
    ("ip_address",   re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")),
    ("date_of_birth", re.compile(
        r"\b(?:DOB|date\s+of\s+birth|born\s+on|birthday)\s*[:\-]?\s*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b", re.I)),
    ("passport",     re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")),
]


def _detect_pii_regex(text: str) -> List[str]:
    """Fast regex-based PII detection. Returns list of entity type strings found."""
    found = []
    for entity_type, pattern in _PII_REGEX_PATTERNS:
        if pattern.search(text):
            found.append(entity_type)
    return found


def _detect_pii_presidio(text: str) -> List[str]:
    """Presidio-based PII detection. Falls back to empty list if not installed."""
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        return list({r.entity_type for r in results})
    except ImportError:
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def detect_exfiltration(
    text: str,
    use_presidio: bool = False,
    content_length_chars: Optional[int] = None,
    session_avg_length: Optional[int] = None,
) -> ExfiltrationResult:
    """
    Scan text for data exfiltration signals.

    Args:
        text:                  The sanitized input text to analyze.
        use_presidio:          Use Microsoft Presidio for PII detection if available.
        content_length_chars:  Length of content in current request (for volume check).
        session_avg_length:    Average content length for this session (for anomaly).

    Returns:
        ExfiltrationResult with score, flags, and found pii_entities.
    """
    if not text:
        return ExfiltrationResult(score=0.0)

    flags: List[str] = []
    pii_entities: List[str] = []
    signal_scores: List[float] = []

    # 1. Credential pattern scan
    for name, pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            flags.append(f"credential:{name}")
            signal_scores.append(0.85)

    # 2. Exfiltration request patterns
    for name, pattern, score in _EXFILTRATION_REQUEST_PATTERNS:
        if pattern.search(text):
            flags.append(f"exfil_request:{name}")
            signal_scores.append(score)

    # 3. PII detection
    if use_presidio:
        pii_entities = _detect_pii_presidio(text)
    if not pii_entities:
        pii_entities = _detect_pii_regex(text)

    if pii_entities:
        flags.append(f"pii_detected:{','.join(pii_entities)}")
        # PII in input isn't automatically exfiltration — lower score signal
        signal_scores.append(0.45)

    # 4. Volume anomaly (if session data provided)
    if content_length_chars is not None and session_avg_length is not None:
        if session_avg_length > 0:
            ratio = content_length_chars / session_avg_length
            if ratio > 10:
                flags.append(f"volume_anomaly:{ratio:.1f}x_avg")
                signal_scores.append(0.60)

    if not signal_scores:
        return ExfiltrationResult(score=0.0)

    max_score = max(signal_scores)
    multi_bonus = min(0.10, (len(signal_scores) - 1) * 0.03)
    final_score = min(1.0, max_score + multi_bonus)

    return ExfiltrationResult(
        score=round(final_score, 4),
        flags=flags,
        pii_entities=pii_entities,
    )
