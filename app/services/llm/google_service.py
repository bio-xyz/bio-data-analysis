from typing import Dict, Type, TypeVar

import instructor
from google import genai
from pydantic import BaseModel

from app.config import get_logger, settings
from app.models.llm_config import LLMConfig
from app.services.llm.base_llm_service import BaseLLMService
from app.utils import SingletonMeta

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class GoogleService(BaseLLMService, metaclass=SingletonMeta):
    """Google Google LLM service implementation."""

    # Supported Google model patterns
    SUPPORTED_PATTERNS = ["google", "gemini"]

    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY must be set to instantiate Google client")

        client_kwargs = {"api_key": settings.GOOGLE_API_KEY}

        self.client = genai.Client(**client_kwargs)
        logger.info("Google client instantiated successfully")

    @classmethod
    def is_supported(cls, model_name: str) -> bool:
        """
        Check if Google is supported for the given model.

        Args:
            model_name: The name of the model to check.

        Returns:
            bool: True if Google API key is set and model matches Google patterns.
        """
        if not settings.GOOGLE_API_KEY:
            return False

        model_lower = model_name.lower()
        return any(pattern in model_lower for pattern in cls.SUPPORTED_PATTERNS)

    def generate_structured(
        self,
        llm_config: LLMConfig,
        messages: list[Dict[str, str]],
        response_model: Type[T],
        mode: instructor.Mode = instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        **kwargs,
    ) -> T:
        """
        Generate a structured response using instructor.

        Args:
            llm_config: Configuration for the LLM model.
            messages: List of message dictionaries with 'role' and 'content' keys.
            response_model: Pydantic model class for the expected response structure.
            mode: Instructor mode for structured output extraction.
                  Options: GENAI_STRUCTURED_OUTPUTS (default), GENAI_TOOLS.
            **kwargs: Additional Google-specific parameters.

        Returns:
            T: An instance of the response_model with validated data.
        """
        instructor_client = instructor.from_genai(self.client, mode=mode)

        params = {
            "model": llm_config.model_name,
            "messages": messages,
            "response_model": response_model,
            "generation_config": {
                "max_tokens": llm_config.max_tokens,
                **kwargs,
            },
        }

        logger.info(
            f"Calling Google API (structured) with model: {llm_config.model_name}, "
            f"response_model: {response_model.__name__}, mode: {mode}"
        )

        response: T = instructor_client.messages.create(**params)
        return response
