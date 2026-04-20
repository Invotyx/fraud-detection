from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "fraud-detection-integration"
    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://fraud:fraud@localhost:5432/fraud_detection"
    redis_url: str = "redis://localhost:6379/0"

    llm_server_url: str = "http://localhost:8001"
    llm_request_timeout: int = 10

    api_key_header: str = "X-API-Key"
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"

    risk_allow_threshold: float = 0.3
    risk_review_threshold: float = 0.7

    hitl_escalation_timeout_seconds: int = 3600

    rate_limit_per_minute: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
