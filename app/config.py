from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    DATABASE_URL: str = "sqlite:///./eligibility.db"
    CLINICALTRIALS_BASE_URL: str = "https://clinicaltrials.gov/api/v2"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
