from typing import Dict

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from app.config import get_logger, settings
from app.models.llm_config import LLMConfig
from app.services.llm.base_llm_service import BaseLLMService
from app.utils import SingletonMeta

logger = get_logger(__name__)


class OpenAIService(BaseLLMService, metaclass=SingletonMeta):
    """OpenAI LLM service implementation."""

    # Supported OpenAI model patterns
    SUPPORTED_PATTERNS = ["gpt", "openai", "o1", "o3", "text-davinci"]

    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set to instantiate OpenAI client")

        client_kwargs = {"api_key": settings.OPENAI_API_KEY}
        if settings.OPENAI_CUSTOM_BASE_URL:
            client_kwargs["base_url"] = settings.OPENAI_CUSTOM_BASE_URL

        self.client = OpenAI(**client_kwargs)
        logger.info("OpenAI client instantiated successfully")

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

    def generate_response(
        self,
        llm_config: LLMConfig,
        messages: list[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate a response using OpenAI API.

        Args:
            llm_config: Configuration for the LLM model.
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional OpenAI-specific parameters (e.g., top_p, frequency_penalty).

        Returns:
            str: The generated response text.
        """
        params = {
            "model": llm_config.model_name,
            "max_completion_tokens": llm_config.max_tokens,
            "messages": messages,
        }

        params.update(kwargs)

        logger.info(f"Calling OpenAI API with model: {llm_config.model_name}")
        response: ChatCompletion = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
