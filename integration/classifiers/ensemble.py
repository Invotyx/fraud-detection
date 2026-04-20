"""
Fraud Model Ensemble
---------------------
Runs existing/future fraud detection models as parallel async signals,
feeding their scores into the Risk Aggregation Engine.

Design:
- Each model is registered as an EnsembleModel with a name, weight, and scorer callable
- All models run concurrently via asyncio.gather with a per-model timeout
- If a model fails or times out, it returns 0.0 with a `not_implemented` flag
- Placeholder models are provided until real models are trained

Usage:
    from classifiers.ensemble import run_ensemble
    scores = await run_ensemble(features)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelScore:
    model_name: str
    score: float            # 0.0 – 1.0
    latency_ms: float
    not_implemented: bool = False
    error: Optional[str] = None


@dataclass
class EnsembleResult:
    scores: List[ModelScore]
    combined_score: float       # weighted average across available models
    unavailable_models: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class EnsembleModel:
    name: str
    weight: float
    scorer: Callable[[Dict[str, Any]], float]
    timeout_seconds: float = 2.0


# ---------------------------------------------------------------------------
# Placeholder scorers (return 0.0 until real models are deployed)
# ---------------------------------------------------------------------------

def _placeholder_scorer(features: Dict[str, Any]) -> float:
    """Placeholder scorer — returns 0.0 (not_implemented)."""
    raise NotImplementedError("Model not yet deployed")


def _rule_based_url_scorer(features: Dict[str, Any]) -> float:
    """
    Simple rule-based URL risk scorer using pre-computed URL features.
    Uses url_risk_score passed directly in features if available.
    """
    return float(features.get("url_risk_score", 0.0))


def _keyword_fraud_scorer(features: Dict[str, Any]) -> float:
    """
    Basic keyword-based fraud signal scorer.
    Checks for known fraud-related keywords in content.
    """
    content: str = features.get("content", "")
    if not content:
        return 0.0

    fraud_keywords = [
        "urgent", "immediate action", "verify your account",
        "click here now", "limited time", "you have won",
        "suspended", "confirm your identity", "unauthorized access",
        "prince", "inheritance", "wire transfer", "gift card",
    ]
    hits = sum(1 for kw in fraud_keywords if kw.lower() in content.lower())
    return min(1.0, hits * 0.12)


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------

DEFAULT_MODELS: List[EnsembleModel] = [
    EnsembleModel(
        name="rule_url_risk",
        weight=0.30,
        scorer=_rule_based_url_scorer,
        timeout_seconds=1.0,
    ),
    EnsembleModel(
        name="keyword_fraud",
        weight=0.30,
        scorer=_keyword_fraud_scorer,
        timeout_seconds=1.0,
    ),
    EnsembleModel(
        name="xgboost_transaction",
        weight=0.40,
        scorer=_placeholder_scorer,      # placeholder until trained
        timeout_seconds=2.0,
    ),
]


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

async def _run_model(
    model: EnsembleModel,
    features: Dict[str, Any],
) -> ModelScore:
    """Run a single model scorer with timeout, catching all errors."""
    start = time.monotonic()
    try:
        loop = asyncio.get_event_loop()
        # Run sync scorer in executor to avoid blocking event loop
        score = await asyncio.wait_for(
            loop.run_in_executor(None, model.scorer, features),
            timeout=model.timeout_seconds,
        )
        score = max(0.0, min(1.0, float(score)))
        latency = (time.monotonic() - start) * 1000
        return ModelScore(
            model_name=model.name,
            score=score,
            latency_ms=round(latency, 2),
        )
    except NotImplementedError:
        latency = (time.monotonic() - start) * 1000
        return ModelScore(
            model_name=model.name,
            score=0.0,
            latency_ms=round(latency, 2),
            not_implemented=True,
        )
    except asyncio.TimeoutError:
        latency = (time.monotonic() - start) * 1000
        return ModelScore(
            model_name=model.name,
            score=0.0,
            latency_ms=round(latency, 2),
            error="timeout",
        )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return ModelScore(
            model_name=model.name,
            score=0.0,
            latency_ms=round(latency, 2),
            error=str(exc)[:80],
        )


async def run_ensemble(
    features: Dict[str, Any],
    models: Optional[List[EnsembleModel]] = None,
) -> EnsembleResult:
    """
    Run all ensemble models concurrently and return combined score.

    Args:
        features: Dict of extracted features available to all models.
                  Expected keys (optional): content, url_risk_score, session_id.
        models:   Model list override (uses DEFAULT_MODELS if None).

    Returns:
        EnsembleResult with per-model scores and weighted combined score.
    """
    model_list = models if models is not None else DEFAULT_MODELS
    model_scores = await asyncio.gather(*[_run_model(m, features) for m in model_list])

    # Weighted average over available (non-error, non-placeholder) models
    total_weight = 0.0
    weighted_sum = 0.0
    unavailable: List[str] = []

    model_map = {m.name: m for m in model_list}
    for ms in model_scores:
        model_def = model_map.get(ms.model_name)
        weight = model_def.weight if model_def else 1.0

        if ms.not_implemented or ms.error:
            unavailable.append(ms.model_name)
            continue  # exclude from scoring

        weighted_sum += ms.score * weight
        total_weight += weight

    combined = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    combined = round(min(1.0, max(0.0, combined)), 6)

    return EnsembleResult(
        scores=list(model_scores),
        combined_score=combined,
        unavailable_models=unavailable,
    )
