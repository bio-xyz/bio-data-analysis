import os
from typing import Optional

from dotenv import load_dotenv

from app.models.llm_config import LLMConfig

load_dotenv()


class Settings:
    """Application settings."""

    # Security Configuration
    API_KEY: Optional[str] = os.getenv("API_KEY")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_CUSTOM_BASE_URL: Optional[str] = os.getenv("OPENAI_CUSTOM_BASE_URL")

    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_CUSTOM_BASE_URL: Optional[str] = os.getenv("ANTHROPIC_CUSTOM_BASE_URL")

    # Default LLM Config
    DEFAULT_LLM: LLMConfig = LLMConfig(
        provider=os.getenv("DEFAULT_PROVIDER", "openai"),
        model_name=os.getenv("DEFAULT_MODEL", "gpt-5"),
    )

    # Plan Generation LLM Config
    PLAN_GENERATION_LLM: LLMConfig = LLMConfig(
        provider=os.getenv("PLAN_GENERATION_PROVIDER", DEFAULT_LLM.provider),
        model_name=os.getenv("PLAN_GENERATION_MODEL", DEFAULT_LLM.model_name),
    )

    # Code Generation LLM Config
    CODE_GENERATION_LLM: LLMConfig = LLMConfig(
        provider=os.getenv("CODE_GENERATION_PROVIDER", DEFAULT_LLM.provider),
        model_name=os.getenv("CODE_GENERATION_MODEL", DEFAULT_LLM.model_name),
    )

    # Response Generation LLM Config
    RESPONSE_GENERATION_LLM: LLMConfig = LLMConfig(
        provider=os.getenv("RESPONSE_GENERATION_PROVIDER", DEFAULT_LLM.provider),
        model_name=os.getenv("RESPONSE_GENERATION_MODEL", DEFAULT_LLM.model_name),
    )

    DEFAULT_WORKING_DIRECTORY: str = os.getenv(
        "DEFAULT_WORKING_DIRECTORY", "/home/user"
    )
    DEFAULT_DATA_DIRECTORY: str = os.getenv("DEFAULT_DATA_DIRECTORY", "/home/user/data")

    # File Upload Configuration
    MAX_FILE_SIZE: int = (
        int(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024
    )  # Convert MB to bytes (default: 100MB)

    # Agent Configuration
    CODE_GENERATION_MAX_RETRIES: int = int(
        os.getenv("CODE_GENERATION_MAX_RETRIES", "3")
    )

    # Task Tracking Configuration
    TASK_CLEANUP_INTERVAL_SECONDS: int = int(
        os.getenv("TASK_CLEANUP_INTERVAL_SECONDS", "60")
    )
    TASK_EXPIRY_SECONDS: int = int(os.getenv("TASK_EXPIRY_SECONDS", "300"))


settings = Settings()
