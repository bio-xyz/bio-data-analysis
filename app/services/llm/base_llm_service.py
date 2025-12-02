from abc import ABC, abstractmethod
from typing import Dict

from app.models.llm_config import LLMConfig


class BaseLLMService(ABC):
    """Base interface for LLM service implementations."""

    @classmethod
    @abstractmethod
    def is_supported(cls, model_name: str) -> bool:
        """
        Check if the service is supported for the given model.

        Args:
            model_name: The name of the model to check.

        Returns:
            bool: True if the service is configured and supports the model.
        """
        pass

    @abstractmethod
    def generate_response(
        self,
        llm_config: LLMConfig,
        messages: list[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            llm_config: Configuration for the LLM model.
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters.

        Returns:
            str: The generated response text.
        """
        pass
