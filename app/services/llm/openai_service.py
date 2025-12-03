from typing import Dict, Type, TypeVar

import instructor
from openai import OpenAI
from pydantic import BaseModel

from app.config import get_logger, settings
from app.models.llm_config import LLMConfig
from app.services.llm.base_llm_service import BaseLLMService
from app.utils import SingletonMeta

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


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

    def generate_structured(
        self,
        llm_config: LLMConfig,
        messages: list[Dict[str, str]],
        response_model: Type[T],
        mode: instructor.Mode = instructor.Mode.JSON,
        **kwargs,
    ) -> T:
        """
        Generate a structured response using instructor.

        Args:
            llm_config: Configuration for the LLM model.
            messages: List of message dictionaries with 'role' and 'content' keys.
            response_model: Pydantic model class for the expected response structure.
            mode: Instructor mode for structured output extraction.
                  Options: TOOLS, JSON (default), MD_JSON, FUNCTIONS.
            **kwargs: Additional OpenAI-specific parameters.

        Returns:
            T: An instance of the response_model with validated data.
        """
        # Create a new instructor client with the specified mode
        instructor_client = instructor.from_openai(self.client, mode=mode)

        params = {
            "model": llm_config.model_name,
            "max_completion_tokens": llm_config.max_tokens,
            "messages": messages,
            "response_model": response_model,
        }

        params.update(kwargs)

        logger.info(
            f"Calling OpenAI API (structured) with model: {llm_config.model_name}, "
            f"response_model: {response_model.__name__}, mode: {mode}"
        )

        response: T = instructor_client.chat.completions.create(**params)
        return response
