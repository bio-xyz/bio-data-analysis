from typing import Any, Dict, Optional

from openai import OpenAI

from app.config import get_logger, settings
from app.services.llm.base_llm_service import BaseLLMService

logger = get_logger(__name__)


class OpenAIService(BaseLLMService):
    """OpenAI LLM service implementation."""

    # Shared client across all instances (class-level singleton)
    _client: Optional[OpenAI] = None

    # Supported OpenAI model patterns
    SUPPORTED_PATTERNS = ["gpt", "openai", "o1", "o3", "text-davinci"]

    def __init__(self, model_name: str):
        """
        Initialize the OpenAI service.

        Args:
            model_name: The name of the OpenAI model to use.
        """
        super().__init__(model_name)
        logger.info(f"OpenAI service initialized with model: {model_name}")

    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        """
        Check if OpenAI is supported for the given model.

        Args:
            model_name: The name of the model to check.

        Returns:
            bool: True if OpenAI API key is set and model matches OpenAI patterns.
        """
        if not settings.OPENAI_API_KEY:
            return False

        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in cls.SUPPORTED_PATTERNS)

    def instantiate_client(self) -> Any:
        """
        Instantiate and return the OpenAI API client.
        Client is shared across all instances of this class.

        Returns:
            The initialized OpenAI client.

        Raises:
            ImportError: If the openai package is not installed.
            ValueError: If the OPENAI_API_KEY is not configured.
        """
        if OpenAIService._client is not None:
            return OpenAIService._client

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                f"Model '{self.model_name}' requires OPENAI_API_KEY to be set"
            )

        client_kwargs = {"api_key": settings.OPENAI_API_KEY}
        if settings.OPENAI_CUSTOM_BASE_URL:
            client_kwargs["base_url"] = settings.OPENAI_CUSTOM_BASE_URL

        OpenAIService._client = OpenAI(**client_kwargs)
        logger.info("OpenAI client instantiated successfully")
        return OpenAIService._client

    def generate_response(
        self,
        messages: list[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate a response using OpenAI API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional OpenAI-specific parameters (e.g., top_p, frequency_penalty).

        Returns:
            str: The generated response text.
        """
        if self._client is None:
            self._client = self.instantiate_client()

        params = {
            "model": self.model_name,
            "messages": messages,
        }

        params.update(kwargs)

        logger.info(f"Calling OpenAI API with model: {self.model_name}")
        response = self._client.chat.completions.create(**params)
        return response.choices[0].message.content
