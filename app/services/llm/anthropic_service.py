from typing import Dict

from anthropic import Anthropic
from anthropic.types.message import Message

from app.config import get_logger, settings
from app.models.llm_config import LLMConfig
from app.services.llm.base_llm_service import BaseLLMService
from app.utils import SingletonMeta

logger = get_logger(__name__)


class AnthropicService(BaseLLMService, metaclass=SingletonMeta):
    """Anthropic LLM service implementation."""

    # Supported Anthropic model patterns
    SUPPORTED_PATTERNS = ["claude", "anthropic"]

    def __init__(self):
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY must be set to instantiate Anthropic client"
            )

        client_kwargs = {"api_key": settings.ANTHROPIC_API_KEY}
        if settings.ANTHROPIC_CUSTOM_BASE_URL:
            client_kwargs["base_url"] = settings.ANTHROPIC_CUSTOM_BASE_URL

        self.client = Anthropic(**client_kwargs)
        logger.info("Anthropic client instantiated successfully")

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

    def generate_response(
        self,
        llm_config: LLMConfig,
        messages: list[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate a response using Anthropic API.

        Args:
            llm_config: Configuration for the LLM model.
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            str: The generated response text.
        """
        # Anthropic requires system messages to be separate
        system_message = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)

        params = {
            "model": llm_config.model_name,
            "max_tokens": llm_config.max_tokens,
            "messages": filtered_messages,
        }

        if system_message:
            params["system"] = system_message

        params.update(kwargs)

        logger.info(f"Calling Anthropic API with model: {llm_config.model_name}")
        response: Message = self.client.messages.create(**params)
        return response.content[0].text
