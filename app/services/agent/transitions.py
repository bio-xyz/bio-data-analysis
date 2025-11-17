"""State transition logic for the agent FST."""

from app.config import get_logger, settings
from app.services.agent.nodes import AgentNode
from app.services.agent.signals import ActionSignal
from app.services.agent.state import AgentState

logger = get_logger(__name__)


def should_continue_after_plan(state: AgentState) -> str:
    """
    Determine next step after plan generation.

    Args:
        state: Current agent state

    Returns:
        Next node to execute: "code_generation" or "analyze"
    """
    action_signal = state.get("action_signal")
    success = state.get("success", True)

    logger.info(f"Plan transition check - Signal: {action_signal}, Success: {success}")

    # If plan failed (missing data), go directly to analyze
    if action_signal == ActionSignal.PLAN_ERROR or not success:
        logger.info("Transition: plan error -> analyze (missing data)")
        return AgentNode.ANALYZE

    # If plan succeeded, proceed to code generation
    if action_signal == ActionSignal.PLAN_COMPLETE:
        logger.info("Transition: plan complete -> code_generation")
        return AgentNode.CODE_GENERATION

    # Default: proceed to code generation
    logger.info("Transition: default -> code_generation")
    return AgentNode.CODE_GENERATION


def should_regenerate_code(state: AgentState) -> str:
    """
    Determine if code should be regenerated after execution error.

    Args:
        state: Current agent state

    Returns:
        Next node to execute: "code_generation" or "analyze"
    """
    action_signal = state.get("action_signal")
    success = state.get("success", True)
    attempts = state.get("code_generation_attempts", 0)
    max_retries = state.get("max_retries", settings.CODE_GENERATION_MAX_RETRIES)

    logger.info(
        f"Execution transition check - Signal: {action_signal}, Success: {success}, "
        f"Attempts: {attempts}/{max_retries}"
    )

    # If execution was successful, move to analyze
    if action_signal == ActionSignal.EXECUTION_SUCCESS and success:
        logger.info("Transition: execution success -> analyze")
        return AgentNode.ANALYZE

    # If execution failed and we haven't exceeded max retries, regenerate code
    if action_signal == ActionSignal.EXECUTION_ERROR and not success:
        if attempts < max_retries:
            logger.info(
                f"Transition: execution error -> regenerate code (attempt {attempts + 1}/{max_retries})"
            )
            return AgentNode.CODE_GENERATION
        else:
            logger.warning(
                f"Transition: max retries reached ({max_retries}) -> analyze with error"
            )
            return AgentNode.ANALYZE

    # Default: move to analyze
    logger.info("Transition: default -> analyze")
    return AgentNode.ANALYZE
