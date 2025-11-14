from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLLMService(ABC):
    """Base interface for LLM service implementations."""

    def __init__(self, model_name: str):
        """
        Initialize the LLM service.

        Args:
            model_name: The name of the model to use.
        """
        self.model_name = model_name

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
    def instantiate_client(self) -> Any:
        """
        Instantiate and return the API client.

        Returns:
            The initialized client.

        Raises:
            ImportError: If the required package is not installed.
            ValueError: If the API key is not configured.
        """
        pass

    @abstractmethod
    def generate_response(
        self,
        messages: list[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate a response using the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters.

        Returns:
            str: The generated response text.
        """
        pass
