"""Transition routing functions for the agent graph.

These functions determine the next node based on the current state and action signals.
"""

from typing import Literal

from app.agent.signals import ActionSignal, AgentNode
from app.agent.state import AgentState
from app.config import get_logger

logger = get_logger(__name__)


def route_after_planning(
    state: AgentState,
) -> Literal[AgentNode.CODE_PLANNING, AgentNode.ANSWERING]:
    """
    Route after PLANNING_NODE.

    Routes to:
    - CODE_PLANNING: Task requires code execution
    - ANSWERING: Direct answer or clarification needed

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    signal = state.get("action_signal")

    if signal == ActionSignal.GENERAL_ANSWER or signal == ActionSignal.CLARIFICATION:
        logger.info("Routing to ANSWERING_NODE (no code needed)")
        return AgentNode.ANSWERING

    logger.info("Routing to CODE_PLANNING_NODE")
    return AgentNode.CODE_PLANNING


def route_after_code_planning(
    state: AgentState,
) -> Literal[AgentNode.CODE_GENERATION, AgentNode.ANSWERING]:
    """
    Route after CODE_PLANNING_NODE.

    Routes to:
    - CODE_GENERATION: Need to generate code for step
    - ANSWERING: Task complete or aborting

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    signal = state.get("action_signal")

    if signal == ActionSignal.TASK_COMPLETED or signal == ActionSignal.TASK_FAILED:
        logger.info("Routing to ANSWERING_NODE (finalize)")
        return AgentNode.ANSWERING

    # ITERATE_CURRENT_STEP or PROCEED_TO_NEXT_STEP both go to code generation
    logger.info("Routing to CODE_GENERATION_NODE")
    return AgentNode.CODE_GENERATION


def route_after_code_generation(
    state: AgentState,
) -> Literal[AgentNode.CODE_EXECUTION]:
    """
    Route after CODE_GENERATION_NODE.

    Always routes to code_execution to run the generated code.

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    logger.info("Routing to CODE_EXECUTION_NODE")
    return AgentNode.CODE_EXECUTION


def route_after_code_execution(
    state: AgentState,
) -> Literal[AgentNode.CODE_PLANNING]:
    """
    Route after CODE_EXECUTION_NODE.

    Always routes back to code_planning to decide next action.
    The LLM in code_planning will decide whether to:
    - Retry the step (on failure)
    - Proceed to next step (on success)
    - Finalize (task complete or giving up)

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    signal = state.get("action_signal", None)
    code_generation_attempts = state.get("code_generation_attempts", 0)

    if signal == ActionSignal.CODE_EXECUTION_SUCCESS:
        logger.info("Code executed successfully, routing to CODE_PLANNING_NODE")
        return AgentNode.CODE_PLANNING
    elif signal == ActionSignal.CODE_EXECUTION_FAILED and code_generation_attempts >= 3:
        logger.info(
            f"Code executed with failure and max attempts reached ({code_generation_attempts}), routing to CODE_PLANNING_NODE for final decision"
        )
        return AgentNode.CODE_PLANNING

    # On failure with attempts left, also go to code_planning to retry
    logger.info("Code execution failed, routing to CODE_GENERATION_NODE for retry")
    return AgentNode.CODE_GENERATION
