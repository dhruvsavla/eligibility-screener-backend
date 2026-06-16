from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── LLM: Anthropic Claude Sonnet ──────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"     # Claude Sonnet
    ANTHROPIC_MAX_TOKENS: int = 4096

    # ── Data / infra ──────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./eligibility.db"
    CLINICALTRIALS_BASE_URL: str = "https://clinicaltrials.gov/api/v2"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── External tool paths ───────────────────────────────────────────────────
    SYNTHEA_JAR_PATH: str = "backend/synthea/synthea.jar"
    OMOP_DIR: str = "backend/omop"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
