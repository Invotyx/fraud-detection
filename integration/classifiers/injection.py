"""
Prompt Injection Classifier
-----------------------------
Detects injection attempts before they reach the LLM.

Two-layer approach:
1. Rule-based (fast path)  — regex patterns for known injection phrases
2. ML classifier (slow path) — DistilBERT-based binary classifier

Also performs indirect injection detection by scanning all text content
in the input (not just the user's direct message).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class InjectionResult:
    score: float               # 0.0 – 1.0
    flags: List[str] = field(default_factory=list)
    rule_match: bool = False   # True → immediate hard block


# ---------------------------------------------------------------------------
# Rule-based patterns
# ---------------------------------------------------------------------------

# Each tuple: (pattern_name, compiled_regex, score_contribution)
INJECTION_RULES: list[tuple[str, re.Pattern, float]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\bignore\s+(previous|all|prior|any)\s+instructions?\b", re.I),
        0.95,
    ),
    (
        "forget_instructions",
        re.compile(
            r"\bforget\s+(everything|your\s+instructions?|all\s+instructions?)\b", re.I),
        0.95,
    ),
    (
        "you_are_now",
        re.compile(r"\byou\s+are\s+now\s+(a|an|the)\b", re.I),
        0.85,
    ),
    (
        "act_as",
        re.compile(r"\bact\s+as\b|\bpretend\s+(you\s+are|to\s+be)\b", re.I),
        0.80,
    ),
    (
        "roleplay_as",
        re.compile(r"\broleplay\s+as\b|\bplay\s+the\s+role\s+of\b", re.I),
        0.75,
    ),
    (
        "jailbreak_keywords",
        re.compile(
            r"\b(jailbreak|developer\s+mode|DAN|do\s+anything\s+now)\b", re.I),
        0.90,
    ),
    (
        "prompt_delimiter_injection",
        re.compile(
            r"(<\|system\|>|<\|user\|>|<\|assistant\|>|\[INST\]|\[/INST\]|"
            r"###\s*system|###\s*instruction|<system>|<\/system>)",
            re.I,
        ),
        0.90,
    ),
    (
        "system_colon",
        re.compile(r"^\s*system\s*:", re.I | re.MULTILINE),
        0.85,
    ),
    (
        "disregard_previous",
        re.compile(r"\bdisregard\s+(previous|prior|all)\b", re.I),
        0.90,
    ),
    (
        "new_instructions",
        re.compile(r"\bnew\s+instructions?\s*:", re.I),
        0.80,
    ),
    (
        "override_directive",
        re.compile(
            r"\boverride\s+(your|all|the)\s+(instructions?|directives?|rules?)\b", re.I),
        0.90,
    ),
    (
        "maintenance_mode",
        re.compile(
            r"\b(maintenance\s+mode|admin\s+mode|god\s+mode|debug\s+mode)\b", re.I),
        0.85,
    ),
    (
        "repeat_system_prompt",
        re.compile(
            r"\b(repeat|print|output|reveal|show|display)\s+(your\s+)?(system\s+prompt|instructions?|prompt)\b", re.I),
        0.88,
    ),
    (
        "translate_instructions",
        re.compile(
            r"\btranslate\s+your\s+(system\s+prompt|instructions?)\b", re.I),
        0.85,
    ),
    # ---- Command / OS injection (INT-5) ----
    (
        "command_injection_shell",
        re.compile(
            r"(?:;\s*rm\s+-[rRfF]{1,3}"
            r"|[|&]\s*(?:curl|wget|nc|netcat|bash|sh|powershell)\b"
            r"|\$\([^)]{1,300}\)"
            r"|`[^`]{1,300}`"
            r"|%0[aA]\s*(?:rm|curl|wget|bash|sh))",
            re.I,
        ),
        0.85,
    ),
    (
        "command_injection_code",
        re.compile(
            r"\b(?:os\.system\s*\("
            r"|subprocess\.(?:run|call|Popen|check_output)\s*\("
            r"|eval\s*\("
            r"|exec\s*\("
            r"|__import__\s*\("
            r"|Invoke-Expression\b"
            r"|Start-Process\b"
            r"|powershell(?:\.exe)?\s+-(?:enc|command|c)\b"
            r"|cmd(?:\.exe)?\s+/[ckC])\b",
            re.I,
        ),
        0.88,
    ),
]

# Confidence threshold above which rule match triggers hard block
RULE_HARD_BLOCK_THRESHOLD: float = 0.85


# ---------------------------------------------------------------------------
# ML classifier (lazy-loaded to avoid startup cost)
# ---------------------------------------------------------------------------

_ml_classifier = None
_ml_tokenizer = None


def _load_ml_classifier():
    """Lazy-load a DistilBERT-based injection classifier. Returns None if not available."""
    global _ml_classifier, _ml_tokenizer
    if _ml_classifier is not None:
        return _ml_classifier, _ml_tokenizer
    try:
        from transformers import pipeline  # type: ignore
        pipe = pipeline(
            "text-classification",
            model="JasperLS/gelectra-base-injection",
            truncation=True,
            max_length=512,
        )
        _ml_classifier = pipe
        return pipe, None
    except Exception:
        return None, None


def _ml_score(text: str) -> Optional[float]:
    """
    Run ML classifier. Returns injection probability [0, 1] or None if unavailable.
    """
    clf, _ = _load_ml_classifier()
    if clf is None:
        return None
    try:
        result = clf(text[:512])[0]
        label: str = result["label"].upper()
        ml_score: float = result["score"]
        if "INJECTION" in label or label == "LABEL_1":
            return ml_score
        return 1.0 - ml_score
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def classify_injection(text: str, use_ml: bool = False) -> InjectionResult:
    """
    Run injection classification on text.

    Args:
        text:   The text to classify (should be sanitized before calling).
        use_ml: If True, also run ML classifier for subtle injections.
                Defaults to False (rule layer only) for speed.

    Returns:
        InjectionResult with score and flags.
    """
    flags: list[str] = []
    max_score: float = 0.0
    rule_match: bool = False

    # --- Rule-based layer ---
    for name, pattern, score in INJECTION_RULES:
        match = pattern.search(text)
        if match:
            flags.append(f"rule:{name}:{match.group(0)[:60]}")
            max_score = max(max_score, score)
            if score >= RULE_HARD_BLOCK_THRESHOLD:
                rule_match = True

    # --- ML classifier layer (optional slow path) ---
    if use_ml:
        ml_prob = _ml_score(text)
        if ml_prob is not None and ml_prob > 0.6:
            flags.append(f"ml_classifier:{ml_prob:.3f}")
            max_score = max(max_score, ml_prob)

    return InjectionResult(
        score=min(1.0, max_score),
        flags=flags,
        rule_match=rule_match,
    )


def scan_for_indirect_injection(content: str, use_ml: bool = False) -> InjectionResult:
    """
    Scan document content for indirect injection patterns.
    This is run on the full document body (after sanitization) to catch
    injections embedded in processed content, not just the user's message.
    """
    return classify_injection(content, use_ml=use_ml)
