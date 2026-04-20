"""
Obfuscation / Evasion Detection Classifier
-------------------------------------------
Detects attempts to hide malicious content from classifiers and the LLM using:

1. Encoding obfuscation  — base64, hex escapes, percent-encoding in non-URL context
2. Unicode homoglyphs    — Cyrillic/Greek/Latin-Extended replacing ASCII lookalikes
3. Fullwidth characters  — Ａ-Ｚ / ａ-ｚ substituted for ASCII
4. Mixed-script strings  — Latin + Cyrillic in the same token
5. Leetspeak             — character substitutions like 1→i, 0→o, 3→e
6. Zero-width / invisible— already caught by sanitizer; re-flagged here as obfuscation
"""
from __future__ import annotations

import re
import base64
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ObfuscationResult:
    score: float                        # 0.0 – 1.0
    flags: List[str] = field(default_factory=list)
    decoded_content: List[str] = field(
        default_factory=list)  # decoded payloads found


# ---------------------------------------------------------------------------
# Encoding patterns
# ---------------------------------------------------------------------------

_BASE64_PATTERN = re.compile(
    r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{20,}={0,2})(?![A-Za-z0-9+/=])"
)
_HEX_ESCAPE_PATTERN = re.compile(r"(?:\\x[0-9a-fA-F]{2}){3,}")
_PERCENT_ENCODE_PATTERN = re.compile(r"(?:%[0-9a-fA-F]{2}){3,}")
_HTML_ENTITY_PATTERN = re.compile(r"(?:&#x?[0-9a-fA-F]{2,4};){3,}")

# ---------------------------------------------------------------------------
# Unicode obfuscation
# ---------------------------------------------------------------------------

# Cyrillic lookalikes of Latin letters (common subset)
_CYRILLIC_LATIN_MAP = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "і": "i", "ѕ": "s", "ԁ": "d",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "У": "Y",
}

# Greek lookalikes
_GREEK_LATIN_MAP = {
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k",
    "ν": "n", "ο": "o", "ρ": "r", "τ": "t", "υ": "u",
    "χ": "x", "ω": "w",
}

# Fullwidth ASCII: U+FF01 (！) to U+FF5E (～) corresponds to U+0021–U+007E
_FULLWIDTH_OFFSET = 0xFF01 - 0x0021


def _has_cyrillic_lookalike(text: str) -> Tuple[bool, str]:
    mixed = [ch for ch in text if ch in _CYRILLIC_LATIN_MAP]
    if mixed:
        return True, f"cyrillic_lookalikes:{','.join(set(mixed))}"
    return False, ""


def _has_greek_lookalike(text: str) -> Tuple[bool, str]:
    mixed = [ch for ch in text if ch in _GREEK_LATIN_MAP]
    if mixed:
        return True, f"greek_lookalikes:{','.join(set(mixed))}"
    return False, ""


def _has_fullwidth(text: str) -> Tuple[bool, int]:
    count = sum(1 for ch in text if 0xFF01 <= ord(ch) <= 0xFF5E)
    return count >= 3, count


def _normalize_fullwidth(text: str) -> str:
    """Convert fullwidth characters to their ASCII equivalents."""
    return "".join(
        chr(ord(ch) - _FULLWIDTH_OFFSET)
        if 0xFF01 <= ord(ch) <= 0xFF5E
        else ch
        for ch in text
    )


def _has_mixed_script(text: str) -> bool:
    """Return True if both Latin and Cyrillic scripts appear in meaningful quantities."""
    scripts: set[str] = set()
    for ch in text:
        name = unicodedata.name(ch, "")
        if "LATIN" in name:
            scripts.add("LATIN")
        elif "CYRILLIC" in name:
            scripts.add("CYRILLIC")
    return "LATIN" in scripts and "CYRILLIC" in scripts


# ---------------------------------------------------------------------------
# Zero-width / invisible characters
# ---------------------------------------------------------------------------

_ZERO_WIDTH_CHARS = frozenset([
    "\u200b", "\u200c", "\u200d", "\ufeff",
    "\u00ad", "\u2060", "\u180e", "\u034f",
])


def _has_zero_width(text: str) -> int:
    return sum(1 for ch in text if ch in _ZERO_WIDTH_CHARS)


# ---------------------------------------------------------------------------
# Leetspeak normalizer
# ---------------------------------------------------------------------------

_LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a",
    "5": "s", "7": "t", "8": "b", "@": "a",
    "!": "i", "$": "s", "+": "t",
})

_LEET_INJECTION_PATTERNS = [
    re.compile(r"\b1gn[o0]r[e3]\b", re.I),
    re.compile(r"\b[f\u0192][o0]rg[e3]t\b", re.I),
    re.compile(r"\b[s\$][y\u0443][s\$]t[e3]m\b", re.I),
    re.compile(r"\b[j\u0458][a4][i!]lb[r\u0433][e3][a4]k\b", re.I),
]


def _detect_leetspeak(text: str) -> Tuple[bool, str]:
    for pat in _LEET_INJECTION_PATTERNS:
        if pat.search(text):
            return True, f"leetspeak:{pat.pattern}"
    # Normalize and check for injection keywords
    normalized = text.translate(_LEET_MAP).lower()
    dangerous = ["ignore", "forget", "system", "jailbreak", "override"]
    matches = [w for w in dangerous if w in normalized and w not in text.lower()]
    if matches:
        return True, f"leetspeak_normalized:{','.join(matches)}"
    return False, ""


# ---------------------------------------------------------------------------
# Base64 decoder helper
# ---------------------------------------------------------------------------

def _try_decode_base64(blob: str) -> str | None:
    """Attempt to decode a potential base64 blob. Returns decoded text or None."""
    # Pad if needed
    padded = blob + "=" * (-len(blob) % 4)
    try:
        decoded = base64.b64decode(padded, validate=True).decode(
            "utf-8", errors="ignore")
        # Only return if decoded output is printable text (not binary)
        if decoded and decoded.isprintable():
            return decoded
    except Exception:
        pass
    return None


def _try_decode_hex_escapes(text: str) -> str | None:
    try:
        # Replace \xNN sequences
        return bytes(
            text.encode("raw_unicode_escape")
        ).decode("unicode_escape")
    except Exception:
        return None


def _try_decode_percent(text: str) -> str | None:
    from urllib.parse import unquote
    try:
        decoded = unquote(text)
        if decoded != text:
            return decoded
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def detect_obfuscation(text: str) -> ObfuscationResult:
    """
    Analyze text for obfuscation/evasion signals.

    Returns ObfuscationResult with score in [0, 1] and list of flags.
    """
    if not text:
        return ObfuscationResult(score=0.0)

    flags: List[str] = []
    decoded_payloads: List[str] = []
    signal_scores: List[float] = []

    # 1. Base64 blobs
    for match in _BASE64_PATTERN.finditer(text):
        blob = match.group(1)
        decoded = _try_decode_base64(blob)
        if decoded:
            flags.append(f"base64_blob:{blob[:20]}...")
            decoded_payloads.append(decoded)
            signal_scores.append(0.75)

    # 2. Hex escape sequences
    if _HEX_ESCAPE_PATTERN.search(text):
        decoded = _try_decode_hex_escapes(text)
        flags.append("hex_escape_sequences")
        if decoded:
            decoded_payloads.append(decoded)
        signal_scores.append(0.70)

    # 3. Percent-encoding in non-URL context
    if _PERCENT_ENCODE_PATTERN.search(text):
        decoded = _try_decode_percent(text)
        flags.append("percent_encoding_non_url")
        if decoded:
            decoded_payloads.append(decoded)
        signal_scores.append(0.65)

    # 4. HTML entity encoding chains
    if _HTML_ENTITY_PATTERN.search(text):
        flags.append("html_entity_chain")
        signal_scores.append(0.60)

    # 5. Cyrillic homoglyphs
    has_cyrillic, cyrillic_flag = _has_cyrillic_lookalike(text)
    if has_cyrillic:
        flags.append(cyrillic_flag)
        signal_scores.append(0.80)

    # 6. Greek homoglyphs
    has_greek, greek_flag = _has_greek_lookalike(text)
    if has_greek:
        flags.append(greek_flag)
        signal_scores.append(0.70)

    # 7. Fullwidth characters
    has_fw, fw_count = _has_fullwidth(text)
    if has_fw:
        flags.append(f"fullwidth_chars:{fw_count}")
        # Re-check normalized text for injection after fullwidth conversion
        normalized = _normalize_fullwidth(text)
        if normalized != text:
            decoded_payloads.append(normalized)
        signal_scores.append(0.75)

    # 8. Mixed Latin + Cyrillic script
    if _has_mixed_script(text):
        flags.append("mixed_latin_cyrillic")
        signal_scores.append(0.70)

    # 9. Zero-width characters (re-flag as obfuscation signal)
    zw_count = _has_zero_width(text)
    if zw_count > 0:
        flags.append(f"zero_width_chars:{zw_count}")
        signal_scores.append(0.60)

    # 10. Leetspeak injection patterns
    has_leet, leet_flag = _detect_leetspeak(text)
    if has_leet:
        flags.append(leet_flag)
        signal_scores.append(0.70)

    # Aggregate score: max of individual signals, tempered by count
    if not signal_scores:
        return ObfuscationResult(score=0.0)

    max_score = max(signal_scores)
    # Bonus for multiple concurrent signals (up to +0.15)
    multi_signal_bonus = min(0.15, (len(signal_scores) - 1) * 0.05)
    final_score = min(1.0, max_score + multi_signal_bonus)

    return ObfuscationResult(
        score=round(final_score, 4),
        flags=flags,
        decoded_content=decoded_payloads,
    )
