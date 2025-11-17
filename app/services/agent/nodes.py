"""Node functions for the agent graph."""

from enum import Enum

from app.config import get_logger, settings
from app.services.agent.signals import ActionSignal
from app.services.agent.state import AgentState
from app.services.executor_service import ExecutorService
from app.services.llm import LLMService

logger = get_logger(__name__)


class AgentNode(str, Enum):
    """Agent node names."""

    PLAN = "plan"
    CODE_GENERATION = "code_generation"
    EXECUTION = "execution"
    ANALYZE = "analyze"


def plan_node(state: AgentState) -> dict:
    """
    Generate a plan for the task.

    Args:
        state: Current agent state

    Returns:
        Updated state with plan and next action signal
    """
    logger.info("Plan node: Generating plan...")

    llm_planner = LLMService(llm_config=settings.PLAN_GENERATION_LLM)

    plan = llm_planner.generate_plan(
        task_description=state["task_description"],
        data_files_description=state["data_files_description"],
        uploaded_files=state["uploaded_files"],
    )

    # Check if plan generation was successful
    if not plan.success:
        logger.warning(f"Plan generation failed: {plan.error}")
        return {
            "plan": plan,
            "error": plan.error,
            "success": False,
            "action_signal": ActionSignal.PLAN_ERROR,
        }

    logger.info(f"Plan generated with {len(plan.steps)} steps")

    return {
        "plan": plan,
        "action_signal": ActionSignal.PLAN_COMPLETE,
    }


def code_generation_node(state: AgentState) -> dict:
    """
    Generate code based on the plan.

    Args:
        state: Current agent state

    Returns:
        Updated state with generated code and next action signal
    """
    logger.info("Code generation node: Generating code...")

    llm_code_generator = LLMService(llm_config=settings.CODE_GENERATION_LLM)

    # Build context for code generation (include error if regenerating)
    success = state.get("success", True)
    error = state.get("error")
    previous_code = state.get("generated_code")

    if not success and error:
        logger.info(f"Regenerating code due to previous error: {error}")

    generated_code = llm_code_generator.generate_code(
        task_description=state["task_description"],
        data_files_description=state["data_files_description"],
        uploaded_files=state["uploaded_files"],
        plan=state["plan"],
        previous_code=previous_code if not success else None,
        previous_error=error if not success else None,
    )

    attempts = state.get("code_generation_attempts", 0) + 1
    logger.info(f"Code generated (attempt {attempts})")

    return {
        "generated_code": generated_code,
        "code_generation_attempts": attempts,
        "action_signal": ActionSignal.CODE_GENERATED,
    }


def execution_node(state: AgentState) -> dict:
    """
    Execute the generated code in the sandbox.

    Args:
        state: Current agent state

    Returns:
        Updated state with execution results and next action signal
    """
    logger.info("Execution node: Executing code...")

    executor_service = ExecutorService()
    sandbox_id = state["sandbox_id"]

    try:
        execution_result = executor_service.execute_code(
            sandbox_id, state["generated_code"]
        )

        # Check if execution had errors
        if execution_result.error is not None:
            logger.warning(f"Execution error: {execution_result.error}")
            return {
                "execution_result": execution_result,
                "error": str(execution_result.error),
                "success": False,
                "action_signal": ActionSignal.EXECUTION_ERROR,
            }

        logger.info("Code executed successfully")
        return {
            "execution_result": execution_result,
            "error": None,
            "success": True,
            "action_signal": ActionSignal.EXECUTION_SUCCESS,
        }

    except Exception as e:
        logger.error(f"Execution node exception: {e}")
        return {
            "error": str(e),
            "success": False,
            "action_signal": ActionSignal.EXECUTION_ERROR,
        }


def analyze_node(state: AgentState) -> dict:
    """
    Analyze execution results and generate final task response.

    Args:
        state: Current agent state

    Returns:
        Updated state with task response and completion signal
    """
    logger.info("Analyze node: Generating task response...")

    llm_response_generator = LLMService(llm_config=settings.RESPONSE_GENERATION_LLM)

    # Get execution result, may be None if execution never succeeded
    execution_result = state.get("execution_result")

    if execution_result is None:
        logger.warning(
            "No execution result available - generating response based on error"
        )

    task_response = llm_response_generator.generate_task_response(
        task_description=state["task_description"],
        generated_code=state.get("generated_code", ""),
        execution_result=execution_result,
        success=state.get("success", True),
        error=state.get("error"),
    )

    # Add plan to response
    task_response.plan = state["plan"]

    logger.info("Task response generated")

    return {
        "task_response": task_response,
        "action_signal": ActionSignal.TASK_COMPLETE,
    }
