from typing import Literal

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for an LLM model."""

    provider: Literal["openai", "anthropic"] = Field(
        description="The LLM provider to use"
    )
    model_name: str = Field(description="The model name to use")
