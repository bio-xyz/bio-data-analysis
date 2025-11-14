import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings."""

    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    CODE_GENERATION_MODEL: str = os.getenv("CODE_GENERATION_MODEL", "gpt-5")
    RESPONSE_GENERATION_MODEL: str = os.getenv("RESPONSE_GENERATION_MODEL", "gpt-5")

    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-5")

    DEFAULT_WORKING_DIRECTORY: str = os.getenv(
        "DEFAULT_WORKING_DIRECTORY", "/home/user"
    )
    DEFAULT_DATA_DIRECTORY: str = os.getenv("DEFAULT_DATA_DIRECTORY", "/home/user/data")


settings = Settings()
