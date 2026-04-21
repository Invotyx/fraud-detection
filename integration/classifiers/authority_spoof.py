"""
Authority Spoof Classifier
---------------------------
Detects messages that impersonate banks, government agencies, payment
networks, or official services by combining two universal rules:

Rule 1 — Authority + Link  (score 0.72)
    A message that (a) identifies itself as coming from an authoritative
    entity AND (b) contains a URL asking the recipient to click/visit is
    almost certainly phishing.  Legitimate banks and government agencies
    never send unsolicited links to collect payments or verify identity.

Rule 2 — Sensitive Information Request  (score 0.80)
    Any message that asks the recipient to *reply with* or *call and
    share* sensitive credentials (OTP, PIN, password, account number,
    CNIC/SSN/NIC, CVV, etc.) is social engineering, regardless of the
    claimed sender.

Both rules are language-agnostic at the pattern level and apply globally
— not just to specific countries.

Signals emitted:
    AuthoritySpoofResult.score   — highest triggered rule weight
    AuthoritySpoofResult.flags   — list of triggered rule names
"""
from __future__ import annotations

import os
import re
import yaml
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "classifiers.yaml"
    )
    with open(cfg_path) as fh:
        return yaml.safe_load(fh).get("authority_spoof", {})


_CFG = _load_cfg()

# ---------------------------------------------------------------------------
# Authority entity keywords (institution name or abbreviated title that
# commonly appears in the sender field or message body)
# ---------------------------------------------------------------------------
_AUTHORITY_KEYWORDS: List[str] = _CFG.get("authority_keywords", [
    # Banks / financial
    "bank", "banking", "credit union", "fintech",
    "paypal", "stripe", "visa", "mastercard", "amex", "american express",
    "western union", "moneygram", "wire transfer",
    # Payment apps (global)
    "jazzcash", "easypaisa", "paytm", "gpay", "google pay", "apple pay",
    "samsung pay", "venmo", "cash app", "revolut", "wise", "skrill",
    # Government / authority
    "government", "federal", "ministry", "department of",
    "irs", "hmrc", "income tax", "tax authority", "revenue",
    "police", "cybercrime", "interpol", "fbi", "cia",
    "immigration", "customs", "border", "passport",
    "social security", "ssa", "medicare", "medicaid",
    "court", "tribunal", "warrant", "legal notice", "summons",
    "fine", "penalty", "challan", "challan due",
    "psca", "nadra", "fbr", "ecp", "sbp",          # Pakistan
    "dvla", "hmrc", "nhs",                           # UK
    "ato", "centrelink",                             # Australia
    "cra", "service canada",                        # Canada
    "sebi", "uidai", "aadhaar", "irctc",            # India
    # Utilities / telecom
    "electricity", "gas board", "water board", "telecom",
    "airtel", "jio", "vodafone", "at&t", "verizon", "t-mobile",
])

# Call-to-action verbs that accompany a link in a spoof message
_CTA_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bclick\b", re.I),
    re.compile(r"\btap\b", re.I),
    re.compile(r"\bvisit\b", re.I),
    re.compile(r"\bopen\b.*link", re.I),
    re.compile(r"\bfollow\b.*link", re.I),
    re.compile(r"\bgo to\b", re.I),
    re.compile(r"\bnavigate to\b", re.I),
    re.compile(r"\bpay.*(?:here|now|at|via)\b", re.I),
    re.compile(r"\bverify.*(?:here|now|at|account)\b", re.I),
    re.compile(r"\bconfirm.*(?:here|now|identity|account)\b", re.I),
    re.compile(r"\bsettle.*(?:here|now|at|fine|payment)\b", re.I),
    re.compile(r"\bdownload\b", re.I),
    re.compile(r"\binstall\b", re.I),
]

# URL detection (plain text)
_URL_PATTERN: re.Pattern = re.compile(
    r"https?://[^\s\"'<>()\[\]{}|\\^`]+", re.I
)

# ---------------------------------------------------------------------------
# Sensitive information request patterns
# ---------------------------------------------------------------------------
_SENSITIVE_REQUEST_PATTERNS: List[re.Pattern] = [
    # OTP / verification code
    re.compile(r"\b(?:share|send|provide|give|enter|reply with|tell us).*\b(?:otp|one.time|verification code|auth code)\b", re.I),
    re.compile(r"\botp\b.*\b(?:sms|text|message)\b", re.I),
    re.compile(r"\b(?:do not|don'?t) share.*\botp\b",
               re.I),  # reverse-psychology phrasing
    # PIN / password
    re.compile(r"\b(?:share|send|provide|give|enter|reply with|confirm).*\b(?:pin|password|passcode|secret|passphrase)\b", re.I),
    # Account / card numbers
    re.compile(r"\b(?:share|send|provide|give|reply with|confirm).*\b(?:account number|card number|iban|routing number)\b", re.I),
    # CVV / expiry
    re.compile(
        r"\b(?:cvv|cvc|security code|expiry|expiration).*(?:share|send|provide|enter|reply|confirm)\b", re.I),
    re.compile(
        r"\b(?:share|send|provide|enter|reply with|confirm).*\b(?:cvv|cvc|security code|expiry|expiration)\b", re.I),
    # National ID / SSN
    re.compile(r"\b(?:share|send|provide|give|reply with|confirm).*\b(?:cnic|nic|ssn|social security|national id|passport number|date of birth|dob)\b", re.I),
    # Generic "personal / sensitive / financial details"
    re.compile(r"\b(?:share|send|provide|give|reply with)\b.*\b(?:personal details|sensitive information|financial details|banking details|bank details)\b", re.I),
    # Asking to call a number to give credentials
    re.compile(
        r"\bcall\b.*\b(?:confirm|verify|provide|share|give).*\b(?:account|otp|pin|password|id)\b", re.I),
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class AuthoritySpoofResult:
    score: float                       # 0.0 – 1.0
    flags: List[str] = field(default_factory=list)
    authority_keyword_matched: str = ""
    sensitive_terms_matched: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _contains_authority_keyword(text: str) -> str:
    """Return the first authority keyword found in text, or empty string."""
    lower = text.lower()
    for kw in _AUTHORITY_KEYWORDS:
        if kw in lower:
            return kw
    return ""


def _has_url(text: str) -> bool:
    return bool(_URL_PATTERN.search(text))


def _has_cta(text: str) -> bool:
    return any(p.search(text) for p in _CTA_PATTERNS)


def _sensitive_terms_matched(text: str) -> List[str]:
    return [p.pattern for p in _SENSITIVE_REQUEST_PATTERNS if p.search(text)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_authority_spoof(text: str) -> AuthoritySpoofResult:
    """
    Analyse *text* for authority-spoof signals.

    Rule weights:
        authority_link:            0.72  — claimed authority + URL + CTA
        sensitive_info_request:    0.80  — asks recipient to share credentials
    Score is the max of all triggered rules (not a sum).
    """
    score = 0.0
    flags: List[str] = []
    authority_kw = ""
    sensitive_matched: List[str] = []

    # --- Rule 1: Authority + Link + CTA ---
    authority_kw = _contains_authority_keyword(text)
    if authority_kw and _has_url(text) and _has_cta(text):
        rule_score = float(_CFG.get("score_weights", {}
                                    ).get("authority_link", 0.72))
        score = max(score, rule_score)
        flags.append(f"authority_link:{authority_kw}")

    # --- Rule 2: Sensitive Information Request ---
    sensitive_matched = _sensitive_terms_matched(text)
    if sensitive_matched:
        rule_score = float(_CFG.get("score_weights", {}).get(
            "sensitive_info_request", 0.80))
        score = max(score, rule_score)
        flags.append("sensitive_info_request")

    return AuthoritySpoofResult(
        score=score,
        flags=flags,
        authority_keyword_matched=authority_kw,
        sensitive_terms_matched=sensitive_matched,
    )
