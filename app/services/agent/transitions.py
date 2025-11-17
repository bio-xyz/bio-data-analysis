"""State transition logic for the agent FST."""

from app.config import get_logger, settings
from app.services.agent.signals import ActionSignal
from app.services.agent.state import AgentState

logger = get_logger(__name__)


def should_regenerate_code(state: AgentState) -> str:
    """
    Determine if code should be regenerated after execution error.

    Args:
        state: Current agent state

    Returns:
        Next node to execute: "code_generation" or "analyze"
    """
    action_signal = state.get("action_signal")
    has_error = state.get("has_execution_error", False)
    attempts = state.get("code_generation_attempts", 0)
    max_retries = state.get("max_retries", settings.CODE_GENERATION_MAX_RETRIES)

    logger.info(
        f"Transition check - Signal: {action_signal}, Error: {has_error}, "
        f"Attempts: {attempts}/{max_retries}"
    )

    # If execution was successful, move to analyze
    if action_signal == ActionSignal.EXECUTION_SUCCESS and not has_error:
        logger.info("Transition: execution success -> analyze")
        return "analyze"

    # If execution failed and we haven't exceeded max retries, regenerate code
    if action_signal == ActionSignal.EXECUTION_ERROR and has_error:
        if attempts < max_retries:
            logger.info(
                f"Transition: execution error -> regenerate code (attempt {attempts + 1}/{max_retries})"
            )
            return "code_generation"
        else:
            logger.warning(
                f"Transition: max retries reached ({max_retries}) -> analyze with error"
            )
            return "analyze"

    # Default: move to analyze
    logger.info("Transition: default -> analyze")
    return "analyze"
