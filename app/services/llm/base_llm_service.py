from abc import ABC, abstractmethod
from typing import Dict, Type, TypeVar

from pydantic import BaseModel

from app.models.llm_config import LLMConfig

T = TypeVar("T", bound=BaseModel)


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
    def generate_structured(
        self,
        llm_config: LLMConfig,
        messages: list[Dict[str, str]],
        response_model: Type[T],
        **kwargs,
    ) -> T:
        """
        Generate a structured response using instructor.

        Args:
            llm_config: Configuration for the LLM model.
            messages: List of message dictionaries with 'role' and 'content' keys.
            response_model: Pydantic model class for the expected response structure.
            **kwargs: Additional provider-specific parameters (e.g., instructor mode).

        Returns:
            T: An instance of the response_model with validated data.
        """
        pass
