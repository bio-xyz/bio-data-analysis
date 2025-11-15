import json
import uuid
from typing import Dict, Optional

from e2b_code_interpreter import Execution
from e2b_code_interpreter.models import serialize_results

from app.config import get_logger, settings
from app.models.llm_config import LLMConfig
from app.models.task import AnswerResponse, ArtifactResponse, TaskResponse
from app.prompts import (
    build_code_generation_prompt,
    build_task_response_prompt,
    get_code_generation_system_prompt,
    get_task_response_system_prompt,
)
from app.services.llm.anthropic_service import AnthropicService
from app.services.llm.base_llm_service import BaseLLMService
from app.services.llm.openai_service import OpenAIService

logger = get_logger(__name__)


class LLMService:
    """Service to handle LLM provider selection and interaction."""

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """
        Initialize the LLM provider service.

        Args:
            llm_config: The LLM configuration to use. If not provided, uses DEFAULT_LLM from settings.
        """
        self.llm_config = llm_config or settings.DEFAULT_LLM
        self.service = self._get_service()

    def _get_service(self) -> BaseLLMService:
        """
        Get the appropriate LLM service based on the provider configuration.

        Returns:
            BaseLLMService: The configured LLM service.

        Raises:
            ValueError: If the provider is not supported or not configured properly.
        """
        provider = self.llm_config.provider.lower()
        model_name = self.llm_config.model_name

        if provider == "openai":
            if not OpenAIService.is_supported(model_name):
                raise ValueError(
                    f"OpenAI provider is not properly configured. "
                    f"Please ensure OPENAI_API_KEY is set and correct model name is used."
                )
            logger.info(f"Using OpenAI service for model: {model_name}")
            return OpenAIService(model_name=model_name)

        elif provider == "anthropic":
            if not AnthropicService.is_supported(model_name):
                raise ValueError(
                    f"Anthropic provider is not properly configured. "
                    f"Please ensure ANTHROPIC_API_KEY is set and correct model name is used."
                )
            logger.info(f"Using Anthropic service for model: {model_name}")
            return AnthropicService(model_name=model_name)

        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers: openai, anthropic"
            )

    def _generate_response(
        self,
        messages: list[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Generate a response using the configured LLM provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters.

        Returns:
            str: The generated response text.
        """
        return self.service.generate_response(messages=messages, **kwargs)

    def generate_code(
        self,
        task_description: str,
        data_files_description: str | None = None,
        uploaded_files: list[str] | None = None,
    ) -> str:
        """
        Generate Python code based on the task description and optional data files.

        Args:
            task_description: Description of the task to accomplish
            data_files_description: Optional description of the data files
            uploaded_files: Optional list of uploaded file names

        Returns:
            str: The generated Python code
        """
        logger.info("Generating code for task...")

        # Build the prompt
        system_prompt = get_code_generation_system_prompt()
        user_prompt = build_code_generation_prompt(
            task_description=task_description,
            data_files_description=data_files_description,
            uploaded_files=uploaded_files,
        )

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate code using the LLM
        generated_code = self._generate_response(messages=messages)

        logger.info(f"Generated code length: {len(generated_code)} characters")
        return generated_code

    def generate_task_response(
        self,
        task_description: str,
        generated_code: str,
        execution_result: Execution,
    ) -> TaskResponse:
        """
        Generate a response summarizing the task, code, and execution result.

        Args:
            task_description: Description of the task
            generated_code: The code that was generated
            execution_result: The result of executing the code
        Returns:
            TaskResponse: The response object containing the summary
        """

        logger.info("Generating task response...")

        execution_dictionary = {
            "artifacts": [],
            "logs": execution_result.logs.to_json(),
            "error": execution_result.to_json() if execution_result.error else None,
        }

        serialized_results = serialize_results(execution_result.results)
        mapped_results: Dict[str, str] = {}
        for result in serialized_results:
            result["id"] = str(uuid.uuid4())
            if "png" in result:
                mapped_results[result["id"]] = result["png"]
                result["png"] = " --- IGNORE --- "

            # Remove chart elements to reduce prompt size
            if "chart" in result and "elements" in result["chart"]:
                result["chart"]["elements"] = " --- IGNORE --- "
            execution_dictionary["artifacts"].append(result)

        execution_json = json.dumps(execution_dictionary)

        logger.debug(f"Execution JSON: {execution_json}")

        # Build the prompt
        system_prompt = get_task_response_system_prompt()
        user_prompt = build_task_response_prompt(
            task_description=task_description,
            generated_code=generated_code,
            execution_json=execution_json,
        )

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate response using the LLM
        response_text = self._generate_response(messages=messages)
        logger.debug(f"LLM Response Text: {response_text}")

        # Parse JSON response
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            # Fallback response
            return TaskResponse(
                answer=AnswerResponse(
                    summary="Task executed",
                    details=["Unable to parse LLM response"],
                ),
                artifacts=[],
            )

        # Build AnswerResponse
        answer = AnswerResponse(
            summary=response_data.get("summary", ""),
            details=response_data.get("details", []),
        )

        artifacts = []
        for artifact_data in response_data.get("artifacts", []):
            content = ""
            if artifact_data.get("id"):
                content = mapped_results.get(artifact_data["id"], "")

            artifact = ArtifactResponse(
                description=artifact_data.get("description"),
                type=artifact_data.get("type") or "unknown",
                filename=artifact_data.get("filename"),
                path=artifact_data.get("path"),
                id=artifact_data.get("id") or str(uuid.uuid4()),
                content=content,
            )
            artifacts.append(artifact)

        logger.info(f"Generated response with {len(artifacts)} artifacts")

        return TaskResponse(answer=answer, artifacts=artifacts)
