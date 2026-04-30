from __future__ import annotations

import re
from functools import lru_cache
from urllib.parse import urlparse

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Private / loopback ranges that must not be reachable from production
_PRIVATE_HOST_RE = re.compile(
    r"^(?:localhost"
    r"|127(?:\.\d+){3}"
    r"|10(?:\.\d+){3}"
    r"|192\.168(?:\.\d+){2}"
    r"|172\.(?:1[6-9]|2\d|3[01])(?:\.\d+){2}"
    r"|169\.254(?:\.\d+){2}"
    r"|::1"
    r"|fd[0-9a-fA-F]{2}:)",
    re.I,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    app_name: str = "fraud-detection-integration"
    app_env: str = "development"
    log_level: str = "INFO"
    api_title: str = "Fraud Detection Integration API"
    api_version: str = "1.0.0"

    # Database / cache
    database_url: str = "postgresql+asyncpg://fraud:fraud@postgres:5432/fraud_detection"
    redis_url: str = "redis://redis:6379/0"

    # LLM inference server
    llm_server_url: str = "http://localhost:8001"
    llm_request_timeout: int = 120
    llm_endpoint: str = "/v1/chat/completions"
    llm_model_name: str = "fraud-detector"
    # path to system_prompt.txt; empty = use bundled default
    llm_system_prompt_path: str = ""

    # Score blending weights (LLM vs rule-based; must sum to 1.0)
    llm_score_weight: float = 0.6
    rule_score_weight: float = 0.4

    # Authentication
    api_key_header: str = "X-API-Key"
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"

    # Risk thresholds (also set in configs/thresholds.yaml)
    risk_allow_threshold: float = 0.3
    risk_review_threshold: float = 0.7

    # Pipeline — must be > llm_request_timeout to allow LLM to respond
    pipeline_timeout_seconds: float = 130.0

    # Inline heuristic scores used before full classifiers run
    url_risk_suspicious_score: float = 0.5
    url_risk_default_score: float = 0.05
    injection_encoding_anomaly_score: float = 0.6

    # CORS — comma-separated origins; "*" allows all
    cors_allowed_origins: str = "*"

    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_window_seconds: float = 60.0

    # Human-in-the-Loop (HITL)
    hitl_escalation_timeout_seconds: int = 3600
    hitl_queue_default_limit: int = 20

    # Guard model pre-filter (pipeline.py Step 0)
    guard_prefilter_score_threshold: float = 0.90

    # LLM inference parameters
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0

    # Audit log — content truncation length for storage
    audit_log_content_max_length: int = 500

    # Input validation — maximum allowed content length (characters)
    max_content_length: int = 50_000

    # Session risk accumulation (classifiers/session_risk.py)
    session_risk_window_seconds: int = 3600
    session_risk_accumulate_threshold: float = 0.40
    session_risk_escalate_threshold: float = 1.20

    # Sanitizer — base64 decode heuristics
    sanitizer_b64_printable_ratio: float = 0.70
    sanitizer_b64_min_decoded_length: int = 10

    # RAG — fraud pattern retrieval
    rag_enabled: bool = True

    @model_validator(mode="after")
    def _validate_llm_ssrf(self) -> "Settings":
        """
        Reject private/loopback LLM server URLs in non-development environments.
        Prevents SSRF via a misconfigured or maliciously set llm_server_url.
        """
        if self.app_env == "development":
            return self
        host = urlparse(self.llm_server_url).hostname or ""
        if _PRIVATE_HOST_RE.match(host):
            raise ValueError(
                f"llm_server_url '{self.llm_server_url}' resolves to a private or "
                "loopback address, which is not permitted outside development. "
                "Set APP_ENV=development to allow local endpoints."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
