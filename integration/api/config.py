from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Application
    app_name: str = "fraud-detection-integration"
    app_env: str = "development"
    log_level: str = "INFO"
    api_title: str = "Fraud Detection Integration API"
    api_version: str = "1.0.0"

    # Database / cache
    database_url: str = "postgresql+asyncpg://fraud:fraud@localhost:5432/fraud_detection"
    redis_url: str = "redis://localhost:6379/0"

    # LLM inference server
    llm_server_url: str = "http://localhost:8001"
    llm_request_timeout: int = 10
    llm_endpoint: str = "/v1/chat/completions"
    llm_model_name: str = "fraud-detector-v1"
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

    # Pipeline
    pipeline_timeout_seconds: float = 5.0

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

    # RAG — fraud pattern retrieval
    rag_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
