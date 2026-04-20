"""
Risk Scoring Aggregation Engine
---------------------------------
Combines signals from all classifiers and the LLM into a unified risk score.

Rules:
- Weighted sum of per-parameter scores (weights from configs/risk_weights.yaml)
- Hard override: any single parameter > 0.90 → BLOCK
- Hard override: URL blocklist match, injection rule match → BLOCK
- Decision thresholds: allow < 0.30, 0.30 <= review < 0.70, block >= 0.70
  (thresholds from configs/thresholds.yaml)
"""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Optional

from api.schemas import Decision

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def _load_yaml(filename: str) -> dict:
    path = os.path.join(_CONFIG_DIR, filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_weights() -> Dict[str, float]:
    data = _load_yaml("risk_weights.yaml")
    return data.get("weights", data)


def _load_thresholds() -> dict:
    return _load_yaml("thresholds.yaml")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AggregationResult:
    unified_risk_score: float
    decision: Decision
    parameter_scores: Dict[str, float] = field(default_factory=dict)
    hard_override: bool = False
    override_reason: str = ""


# ---------------------------------------------------------------------------
# Hard override flags
# ---------------------------------------------------------------------------

@dataclass
class HardOverrideFlags:
    url_blocklist_match: bool = False
    injection_rule_match: bool = False


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

def aggregate(
    parameter_scores: Dict[str, float],
    hard_flags: Optional[HardOverrideFlags] = None,
    weights_override: Optional[Dict[str, float]] = None,
    thresholds_override: Optional[dict] = None,
) -> AggregationResult:
    """
    Compute unified risk score and decision from per-parameter scores.

    Args:
        parameter_scores:   Dict mapping parameter name → score in [0, 1].
                            Expected keys: url_domain_risk, fraud_intent,
                            prompt_injection, context_deviation,
                            data_exfiltration, obfuscation_evasion,
                            unauthorized_action.
        hard_flags:         Explicit hard-block flags (blocklist hit, rule match).
        weights_override:   Override weights (for testing).
        thresholds_override: Override thresholds (for testing).

    Returns:
        AggregationResult with unified score and Decision.
    """
    weights = weights_override if weights_override is not None else _load_weights()
    thresholds = thresholds_override if thresholds_override is not None else _load_thresholds()

    allow_threshold: float = thresholds.get("allow", 0.30)
    review_threshold: float = thresholds.get("review", 0.70)
    hard_block_single: float = thresholds.get("hard_block_single_param", 0.90)

    # 1. Hard override: URL blocklist or injection rule match
    if hard_flags is not None:
        if hard_flags.url_blocklist_match:
            return AggregationResult(
                unified_risk_score=1.0,
                decision=Decision.BLOCK,
                parameter_scores=parameter_scores,
                hard_override=True,
                override_reason="url_blocklist_match",
            )
        if hard_flags.injection_rule_match:
            return AggregationResult(
                unified_risk_score=1.0,
                decision=Decision.BLOCK,
                parameter_scores=parameter_scores,
                hard_override=True,
                override_reason="injection_rule_match",
            )

    # 2. Hard override: any single parameter > hard_block_single threshold
    for param, score in parameter_scores.items():
        if score > hard_block_single:
            return AggregationResult(
                unified_risk_score=min(1.0, score),
                decision=Decision.BLOCK,
                parameter_scores=parameter_scores,
                hard_override=True,
                override_reason=f"single_param_hard_block:{param}={score:.3f}",
            )

    # 3. Weighted aggregation
    total_weight = 0.0
    weighted_sum = 0.0
    for param, weight in weights.items():
        score = parameter_scores.get(param, 0.0)
        # Clamp individual scores to [0, 1]
        score = max(0.0, min(1.0, score))
        weighted_sum += score * weight
        total_weight += weight

    # Normalize in case weights don't sum to exactly 1.0
    if total_weight > 0:
        unified_score = weighted_sum / total_weight
    else:
        unified_score = 0.0

    unified_score = round(min(1.0, max(0.0, unified_score)), 6)

    # 4. Decision
    if unified_score < allow_threshold:
        decision = Decision.ALLOW
    elif unified_score < review_threshold:
        decision = Decision.REVIEW
    else:
        decision = Decision.BLOCK

    return AggregationResult(
        unified_risk_score=unified_score,
        decision=decision,
        parameter_scores=parameter_scores,
        hard_override=False,
    )
