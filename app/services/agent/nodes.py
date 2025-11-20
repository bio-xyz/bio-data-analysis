"""Node functions for the agent graph."""

from enum import Enum

from app.config import get_logger, settings
from app.services.agent.signals import ActionSignal
from app.services.agent.state import AgentState
from app.services.executor_service import ExecutorService
from app.services.llm import LLMService
from app.utils.nb_builder import NotebookBuilder

logger = get_logger(__name__)


class AgentNode(str, Enum):
    """Agent node names."""

    PLAN = "plan"
    CODE_GENERATION = "code_generation"
    EXECUTION = "execution"
    ANALYZE = "analyze"


def _cleanup_context(
    executor_service: ExecutorService, sandbox_id: str, nb_builder: NotebookBuilder
) -> None:
    """
    Recreate sandbox context and clear notebook builder.

    Args:
        executor_service: Executor service instance
        sandbox_id: Sandbox identifier
        nb_builder: Notebook builder instance
    """
    executor_service.create_context(sandbox_id)
    nb_builder.clear()
    logger.info(f"Recreated context for sandbox {sandbox_id}")


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
    nb_builder = state.get("notebook_builder", NotebookBuilder())
    generated_code = state.get("generated_code", "")

    # Add the generated code as a code cell
    nb_builder.add_code(generated_code)

    try:
        execution_result = executor_service.execute_code(sandbox_id, generated_code)
    except Exception as e:
        logger.error(f"Execution failed with exception: {e}")
        _cleanup_context(executor_service, sandbox_id, nb_builder)
        return {
            "execution_result": None,
            "error": f"Execution failed: {str(e)}",
            "success": False,
            "action_signal": ActionSignal.EXECUTION_ERROR,
            "notebook_builder": nb_builder,
        }

    # Attach execution outputs to the notebook builder
    nb_builder.add_execution(execution_result)

    # Check if execution had errors
    if execution_result.error is not None:
        logger.warning(f"Execution error: {execution_result.error}")
        _cleanup_context(executor_service, sandbox_id, nb_builder)
        return {
            "execution_result": execution_result,
            "error": str(execution_result.error),
            "success": False,
            "action_signal": ActionSignal.EXECUTION_ERROR,
            "notebook_builder": nb_builder,
        }

    logger.info("Code executed successfully")
    return {
        "execution_result": execution_result,
        "error": None,
        "success": True,
        "action_signal": ActionSignal.EXECUTION_SUCCESS,
        "notebook_builder": nb_builder,
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
