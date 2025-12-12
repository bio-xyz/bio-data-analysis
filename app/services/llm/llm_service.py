from typing import Dict, Optional, Type, TypeVar

import instructor
from pydantic import BaseModel

from app.config import get_logger, settings
from app.models.llm_config import LLMConfig
from app.models.structured_outputs import (
    ClarificationResponse,
    CodePlanningDecision,
    ExecutionObserverDecision,
    GeneralAnswerResponse,
    PlanningDecision,
    PythonCode,
    ReflectionDecision,
    StepObservation,
    TaskResponseAnswer,
)
from app.models.task import CompletedStep
from app.prompts import (
    build_code_generation_prompt,
    build_code_planning_prompt,
    build_execution_observer_prompt,
    build_general_answer_prompt,
    build_planning_prompt,
    build_reflection_prompt,
    build_task_clarification_prompt,
    build_task_response_prompt,
    get_code_generation_system_prompt,
    get_code_planning_system_prompt,
    get_execution_observer_system_prompt,
    get_general_answer_system_prompt,
    get_planning_system_prompt,
    get_reflection_system_prompt,
    get_task_clarification_system_prompt,
    get_task_response_system_prompt,
)
from app.services.llm.anthropic_service import AnthropicService
from app.services.llm.base_llm_service import BaseLLMService
from app.services.llm.google_service import GoogleService
from app.services.llm.openai_service import OpenAIService

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


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
            return OpenAIService()

        elif provider == "anthropic":
            if not AnthropicService.is_supported(model_name):
                raise ValueError(
                    f"Anthropic provider is not properly configured. "
                    f"Please ensure ANTHROPIC_API_KEY is set and correct model name is used."
                )
            logger.info(f"Using Anthropic service for model: {model_name}")
            return AnthropicService()

        elif provider == "google":
            if not GoogleService.is_supported(model_name):
                raise ValueError(
                    f"Google provider is not properly configured. "
                    f"Please ensure GOOGLE_API_KEY is set and correct model name is used."
                )
            logger.info(f"Using Google service for model: {model_name}")
            return GoogleService()

        else:
            raise ValueError(
                f"Unsupported provider: {provider}. Supported providers: openai, anthropic, google"
            )

    def _generate_structured(
        self,
        messages: list[Dict[str, str]],
        response_model: Type[T],
        mode: Optional[instructor.Mode] = None,
        **kwargs,
    ) -> T:
        """
        Generate a structured response using instructor.

        This is a unified API that works across both OpenAI and Anthropic providers.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            response_model: Pydantic model class for the expected response structure.
            mode: Optional instructor mode. If not provided, uses provider-specific defaults:
                  - OpenAI: instructor.Mode.TOOLS
                  - Anthropic: instructor.Mode.ANTHROPIC_TOOLS
            **kwargs: Additional provider-specific parameters.

        Returns:
            T: An instance of the response_model with validated data.
        """
        call_kwargs = {**kwargs}
        if mode is not None:
            call_kwargs["mode"] = mode

        return self.service.generate_structured(
            llm_config=self.llm_config,
            messages=messages,
            response_model=response_model,
            **call_kwargs,
        )

    def generate_task_response_answer(
        self,
        task_description: str,
        completed_steps: Optional[list[CompletedStep]] = None,
        failure_reason: Optional[str] = None,
        workdir_contents: Optional[str] = None,
        world_observations: Optional[list[StepObservation]] = None,
    ) -> TaskResponseAnswer:
        """
        Generate a response summarizing the task, code, and execution result.

        Args:
            task_description: Description of the task
            completed_steps: List of completed steps
            failure_reason: Reason for failure if any
            workdir_contents: Contents of the working directory
            world_observations: Refined world observations from reflection node

        Returns:
            TaskResponseAnswer: The response object containing the summary
        """

        logger.info("Generating task response...")

        system_prompt = get_task_response_system_prompt()
        user_prompt = build_task_response_prompt(
            task_description=task_description,
            completed_steps=completed_steps,
            failure_reason=failure_reason,
            workdir_contents=workdir_contents,
            world_observations=world_observations,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_answer: TaskResponseAnswer = self._generate_structured(
            messages=messages,
            response_model=TaskResponseAnswer,
        )

        logger.info(
            f"Generated response with {len(response_answer.artifacts)} artifacts"
        )

        return response_answer

    def generate_planning_decision(
        self,
        task_description: str,
        data_files_description: Optional[str] = None,
        uploaded_files: Optional[list[str]] = None,
    ) -> PlanningDecision:
        """
        Generate planning decision (PLANNING_NODE).

        Decides whether the task requires code execution or can be answered directly.

        Args:
            task_description: Description of the task
            data_files_description: Optional description of data files
            uploaded_files: Optional list of uploaded file names

        Returns:
            PlanningDecision: Planning decision with signal and rationale
        """
        logger.info("Generating planning decision...")

        system_prompt = get_planning_system_prompt()
        user_prompt = build_planning_prompt(
            task_description=task_description,
            data_files_description=data_files_description,
            uploaded_files=uploaded_files,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        decision = self._generate_structured(
            messages=messages,
            response_model=PlanningDecision,
        )
        logger.info(f"Planning decision: {decision.signal}")
        return decision

    def generate_code_planning_decision(
        self,
        task_description: str,
        data_files_description: Optional[str] = None,
        uploaded_files: Optional[list[str]] = None,
        current_step_goal: Optional[str] = None,
        current_step_goal_history: Optional[list[str]] = None,
        current_step_observations: Optional[list[StepObservation]] = None,
        current_step_success: bool = True,
        completed_steps: Optional[list[CompletedStep]] = None,
        world_observations: Optional[list[StepObservation]] = None,
    ) -> CodePlanningDecision:
        """
        Generate code planning decision (CODE_PLANNING_NODE).

        Decides next step: iterate, proceed, or finalize.
        The LLM has full autonomy to decide when to finalize - there's no hard cutoff.

        Args:
            task_description: Description of the task
            data_files_description: Optional description of data files
            uploaded_files: Optional list of uploaded file names
            current_step_goal: Current step goal to be executed
            current_step_goal_history: History of current step goals tried
            current_step_observations: Observations from execution observer for current step (for failure context)
            current_step_success: Whether current step execution was successful
            completed_steps: List of completed steps with results
            world_observations: Refined world observations from reflection node

        Returns:
            CodePlanningDecision: Decision with signal, current_step_goal, current_step_description, reasoning
        """
        logger.info("Generating code planning decision...")

        system_prompt = get_code_planning_system_prompt()
        user_prompt = build_code_planning_prompt(
            task_description=task_description,
            data_files_description=data_files_description,
            uploaded_files=uploaded_files,
            current_step_goal=current_step_goal,
            current_step_goal_history=current_step_goal_history,
            current_step_observations=current_step_observations,
            current_step_success=current_step_success,
            completed_steps=completed_steps,
            world_observations=world_observations,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        decision = self._generate_structured(
            messages=messages,
            response_model=CodePlanningDecision,
        )
        logger.info(f"Code planning decision: {decision.signal}")
        return decision

    def generate_step_code(
        self,
        current_step_goal: str,
        current_step_description: Optional[str] = None,
        data_files_description: Optional[str] = None,
        uploaded_files: Optional[list[str]] = None,
        last_execution_output: Optional[str] = None,
        last_execution_error: Optional[str] = None,
        notebook_code: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> PythonCode:
        """
        Generate code for a specific step (CODE_GENERATION_NODE).

        Args:
            current_step_goal: The goal of the current step
            current_step_description: Optional detailed description of the current step
            data_files_description: Optional description of data files
            uploaded_files: Optional list of uploaded file names
            last_execution_output: Output from last execution
            last_execution_error: Error from last execution
            notebook_code: Code already present in the notebook
            previous_code: Previously generated code for context

        Returns:
            PythonCode: Generated Python code for the step (with code and optional reasoning)
        """
        logger.info(f"Generating code for step: {current_step_goal}")

        system_prompt = get_code_generation_system_prompt()
        user_prompt = build_code_generation_prompt(
            current_step_goal=current_step_goal,
            current_step_description=current_step_description,
            data_files_description=data_files_description,
            uploaded_files=uploaded_files,
            last_execution_output=last_execution_output,
            last_execution_error=last_execution_error,
            notebook_code=notebook_code,
            previous_code=previous_code,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self._generate_structured(
            messages=messages,
            response_model=PythonCode,
        )
        logger.info(f"Generated step code length: {len(result.code)} characters")

        return result

    def generate_clarification_questions(
        self,
        task_description: str,
        task_rationale: str,
    ) -> ClarificationResponse:
        """
        Generate clarification questions.

        Args:
            task_description: Description of the task
            task_rationale: Rationale from planning node

        Returns:
            ClarificationResponse: Clarification questions to ask the user
        """

        logger.info("Generating clarification questions...")

        system_prompt = get_task_clarification_system_prompt()
        user_prompt = build_task_clarification_prompt(
            task_description=task_description,
            task_rationale=task_rationale,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self._generate_structured(
            messages=messages,
            response_model=ClarificationResponse,
        )
        logger.debug(f"Clarification questions response: {result.questions}")

        return result

    def generate_general_answer(
        self,
        task_description: str,
        task_rationale: str,
    ) -> GeneralAnswerResponse:
        """
        Generate a general answer for the task.

        Args:
            task_description: Description of the task
            task_rationale: Rationale from planning node
        Returns:
            GeneralAnswerResponse: General answer to the task
        """

        logger.info("Generating general answer...")

        system_prompt = get_general_answer_system_prompt()
        user_prompt = build_general_answer_prompt(
            task_description=task_description,
            task_rationale=task_rationale,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self._generate_structured(
            messages=messages,
            response_model=GeneralAnswerResponse,
        )
        logger.debug(f"General answer response: {result.answer}")

        return result

    def generate_execution_observations(
        self,
        task_description: str,
        current_step_goal: str,
        current_step_description: Optional[str] = None,
        execution_output: Optional[str] = None,
        execution_error: Optional[str] = None,
    ) -> ExecutionObserverDecision:
        """
        Generate observations from code execution results (EXECUTION_OBSERVER_NODE).

        Args:
            task_description: Description of the overall task
            current_step_goal: Current step being worked on
            current_step_description: Detailed description of the current step
            execution_output: Output from code execution
            execution_error: Error from code execution (if any)

        Returns:
            ExecutionObserverDecision: Observations extracted from execution
        """
        logger.info("Generating execution observations...")

        system_prompt = get_execution_observer_system_prompt()
        user_prompt = build_execution_observer_prompt(
            task_description=task_description,
            current_step_goal=current_step_goal,
            current_step_description=current_step_description,
            execution_output=execution_output,
            execution_error=execution_error,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self._generate_structured(
            messages=messages,
            response_model=ExecutionObserverDecision,
        )
        logger.info(f"Generated {len(result.observations)} observations")

        return result

    def generate_reflection(
        self,
        task_description: str,
        current_step_success: bool,
        current_step_observations: Optional[list[StepObservation]] = None,
        world_observations: Optional[list[StepObservation]] = None,
    ) -> ReflectionDecision:
        """
        Generate refined world observations from reflection (REFLECTION_NODE).

        Merges new observations with existing world observations, deduplicates,
        and refines the observation set.

        Args:
            task_description: Description of the overall task
            current_step_success: Whether the current step succeeded
            current_step_observations: New observations from the current step
            world_observations: Existing world observations to merge with

        Returns:
            ReflectionDecision: Refined rules and data observations
        """
        logger.info("Generating reflection on observations...")

        system_prompt = get_reflection_system_prompt()
        user_prompt = build_reflection_prompt(
            task_description=task_description,
            current_step_success=current_step_success,
            current_step_observations=current_step_observations,
            world_observations=world_observations,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = self._generate_structured(
            messages=messages,
            response_model=ReflectionDecision,
        )
        logger.info(
            f"Reflection generated {len(result.rules)} rules and {len(result.data_observations)} data observations"
        )

        return result
