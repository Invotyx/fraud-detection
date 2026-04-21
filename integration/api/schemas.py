from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Decision(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


# ---------------------------------------------------------------------------
# Per-parameter signal
# ---------------------------------------------------------------------------


class ParameterScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    flag: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# LLM response schema  (also used as the canonical output schema)
# ---------------------------------------------------------------------------


class FraudAnalysisResult(BaseModel):
    url_domain_risk: ParameterScore = Field(
        default_factory=lambda: ParameterScore(score=0.0))
    fraud_intent: ParameterScore = Field(
        default_factory=lambda: ParameterScore(score=0.0))
    prompt_injection: ParameterScore = Field(
        default_factory=lambda: ParameterScore(score=0.0))
    context_deviation: ParameterScore = Field(
        default_factory=lambda: ParameterScore(score=0.0))
    data_exfiltration: ParameterScore = Field(
        default_factory=lambda: ParameterScore(score=0.0))
    obfuscation_evasion: ParameterScore = Field(
        default_factory=lambda: ParameterScore(score=0.0))
    unauthorized_action: ParameterScore = Field(
        default_factory=lambda: ParameterScore(score=0.0))
    authority_spoof: ParameterScore = Field(
        default_factory=lambda: ParameterScore(score=0.0))
    unified_risk_score: float = Field(0.0, ge=0.0, le=1.0)
    decision: Decision = Decision.ALLOW
    explanation: str = ""


# ---------------------------------------------------------------------------
# API request / response
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=50_000)
    session_id: Optional[str] = None
    # declared scope for context deviation checks
    task_scope: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    trace_id: UUID = Field(default_factory=uuid4)
    result: FraudAnalysisResult
    processing_time_ms: int
    hitl_pending: bool = False


# ---------------------------------------------------------------------------
# Sanitizer result
# ---------------------------------------------------------------------------


class SanitizedResult(BaseModel):
    original_hash: str          # SHA-256 of raw input
    sanitized_text: str
    sanitized_hash: str         # SHA-256 of sanitized output
    removed_elements: List[str] = Field(default_factory=list)
    detected_urls: List[str] = Field(default_factory=list)
    encoding_anomalies: List[str] = Field(default_factory=list)
