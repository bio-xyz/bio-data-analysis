import os
from typing import Optional

from dotenv import load_dotenv

from app.models.llm_config import LLMConfig

load_dotenv()


def _get_llm_config(node_name: str, default: LLMConfig) -> LLMConfig:
    """Helper to create LLM config with defaults based on node name."""
    return LLMConfig(
        provider=os.getenv(f"{node_name}_PROVIDER", default.provider),
        model_name=os.getenv(f"{node_name}_MODEL", default.model_name),
        max_tokens=int(os.getenv(f"{node_name}_MAX_TOKENS") or default.max_tokens),
    )


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

    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

    # Default LLM Config
    DEFAULT_LLM: LLMConfig = LLMConfig(
        provider=os.getenv("DEFAULT_PROVIDER", "openai"),
        model_name=os.getenv("DEFAULT_MODEL", "gpt-5"),
        max_tokens=int(os.getenv("DEFAULT_MAX_TOKENS") or 4096),
    )

    # Planning Node LLM Config
    PLANNING_LLM: LLMConfig = _get_llm_config("PLANNING", DEFAULT_LLM)

    # Code planning Node LLM Config
    CODE_PLANNING_LLM: LLMConfig = _get_llm_config("CODE_PLANNING", DEFAULT_LLM)

    # Code generation Node LLM Config
    CODE_GENERATION_LLM: LLMConfig = _get_llm_config("CODE_GENERATION", DEFAULT_LLM)

    # Answering Node LLM Config
    ANSWERING_LLM: LLMConfig = _get_llm_config("ANSWERING", DEFAULT_LLM)

    DEFAULT_WORKING_DIRECTORY: str = os.getenv(
        "DEFAULT_WORKING_DIRECTORY", "/home/user"
    )
    DEFAULT_DATA_DIRECTORY: str = os.getenv("DEFAULT_DATA_DIRECTORY", "/home/user/data")

    # File Upload Configuration
    MAX_FILE_SIZE: int = (
        int(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024
    )  # Convert MB to bytes (default: 100MB)

    # Agent Configuration
    CODE_PLANNING_MAX_STEP_RETRIES: int = int(
        os.getenv("CODE_PLANNING_MAX_STEP_RETRIES", "3")
    )
    CODE_GENERATION_MAX_RETRIES: int = int(
        os.getenv("CODE_GENERATION_MAX_RETRIES", "5")
    )

    # Task Tracking Configuration
    TASK_CLEANUP_INTERVAL_SECONDS: int = int(
        os.getenv("TASK_CLEANUP_INTERVAL_SECONDS", "60")
    )
    TASK_EXPIRY_SECONDS: int = int(os.getenv("TASK_EXPIRY_SECONDS", "300"))

    # Sandbox Configuration
    SANDBOX_DEFAULT_TIMEOUT_SECONDS: int = int(
        os.getenv("SANDBOX_DEFAULT_TIMEOUT_SECONDS", "2400")
    )


settings = Settings()
