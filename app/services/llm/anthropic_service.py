from typing import Any, Dict, Optional

from anthropic import Anthropic

from app.config import get_logger, settings
from app.services.llm.base_llm_service import BaseLLMService

logger = get_logger(__name__)


class AnthropicService(BaseLLMService):
    """Anthropic LLM service implementation."""

    # Shared client across all instances (class-level singleton)
    _client: Optional[Anthropic] = None

    # Supported Anthropic model patterns
    SUPPORTED_PATTERNS = ["claude", "anthropic"]

    def __init__(self, model_name: str):
        """
        Initialize the Anthropic service.

        Args:
            model_name: The name of the Anthropic model to use.
        """
        super().__init__(model_name)
        logger.info(f"Anthropic service initialized with model: {model_name}")

    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        """
        Check if Anthropic is supported for the given model.

        Args:
            model_name: The name of the model to check.

        Returns:
            bool: True if Anthropic API key is set and model matches Anthropic patterns.
        """
        if not settings.ANTHROPIC_API_KEY:
            return False

        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in cls.SUPPORTED_PATTERNS)

    def instantiate_client(self) -> Any:
        """
        Instantiate and return the Anthropic API client.
        Client is shared across all instances of this class.

        Returns:
            The initialized Anthropic client.

        Raises:
            ImportError: If the anthropic package is not installed.
            ValueError: If the ANTHROPIC_API_KEY is not configured.
        """
        if AnthropicService._client is not None:
            return AnthropicService._client

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                f"Model '{self.model_name}' requires ANTHROPIC_API_KEY to be set"
            )

        try:
            AnthropicService._client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info("Anthropic client instantiated successfully")
            return AnthropicService._client
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic models. Install with: pip install anthropic"
            )

    def generate_response(
        self,
        messages: list[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate a response using Anthropic API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            str: The generated response text.
        """
        if self._client is None:
            self._client = self.instantiate_client()

        # Anthropic requires system messages to be separate
        system_message = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)

        params = {
            "model": self.model_name,
            "messages": filtered_messages,
        }

        if system_message:
            params["system"] = system_message

        params.update(kwargs)

        logger.info(f"Calling Anthropic API with model: {self.model_name}")
        response = self._client.messages.create(**params)
        return response.content[0].text
