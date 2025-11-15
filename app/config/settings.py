import os
from typing import Optional

from dotenv import load_dotenv

from app.models.llm_config import LLMConfig

load_dotenv()


class Settings:
    """Application settings."""

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_CUSTOM_BASE_URL: Optional[str] = os.getenv("OPENAI_CUSTOM_BASE_URL")

    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_CUSTOM_BASE_URL: Optional[str] = os.getenv("ANTHROPIC_CUSTOM_BASE_URL")

    # Code Generation LLM Config
    CODE_GENERATION_LLM: LLMConfig = LLMConfig(
        provider=os.getenv("CODE_GENERATION_PROVIDER", "openai"),
        model_name=os.getenv("CODE_GENERATION_MODEL", "gpt-5"),
    )

    # Response Generation LLM Config
    RESPONSE_GENERATION_LLM: LLMConfig = LLMConfig(
        provider=os.getenv("RESPONSE_GENERATION_PROVIDER", "openai"),
        model_name=os.getenv("RESPONSE_GENERATION_MODEL", "gpt-5"),
    )

    # Default LLM Config
    DEFAULT_LLM: LLMConfig = LLMConfig(
        provider=os.getenv("DEFAULT_PROVIDER", "openai"),
        model_name=os.getenv("DEFAULT_MODEL", "gpt-5"),
    )

    DEFAULT_WORKING_DIRECTORY: str = os.getenv(
        "DEFAULT_WORKING_DIRECTORY", "/home/user"
    )
    DEFAULT_DATA_DIRECTORY: str = os.getenv("DEFAULT_DATA_DIRECTORY", "/home/user/data")


settings = Settings()
