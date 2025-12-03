from typing import Dict, Type, TypeVar

import instructor
from anthropic import Anthropic
from pydantic import BaseModel

from app.config import get_logger, settings
from app.models.llm_config import LLMConfig
from app.services.llm.base_llm_service import BaseLLMService
from app.utils import SingletonMeta

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


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

    def generate_structured(
        self,
        llm_config: LLMConfig,
        messages: list[Dict[str, str]],
        response_model: Type[T],
        mode: instructor.Mode = instructor.Mode.ANTHROPIC_JSON,
        **kwargs,
    ) -> T:
        """
        Generate a structured response using instructor.

        Args:
            llm_config: Configuration for the LLM model.
            messages: List of message dictionaries with 'role' and 'content' keys.
            response_model: Pydantic model class for the expected response structure.
            mode: Instructor mode for structured output extraction.
                  Options: ANTHROPIC_TOOLS, ANTHROPIC_JSON (default).
            **kwargs: Additional Anthropic-specific parameters.

        Returns:
            T: An instance of the response_model with validated data.
        """
        instructor_client = instructor.from_anthropic(self.client, mode=mode)

        params = {
            "model": llm_config.model_name,
            "max_tokens": llm_config.max_tokens,
            "messages": messages,
            "response_model": response_model,
            **kwargs,
        }

        logger.info(
            f"Calling Anthropic API (structured) with model: {llm_config.model_name}, "
            f"response_model: {response_model.__name__}, mode: {mode}"
        )

        response: T = instructor_client.messages.create(**params)
        return response
