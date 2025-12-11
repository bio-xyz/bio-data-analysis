"""Transition routing functions for the agent graph.

These functions determine the next node based on the current state and action signals.
"""

from typing import Literal

from app.agent.signals import ActionSignal, AgentNode
from app.agent.state import AgentState
from app.config import get_logger, settings

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
) -> Literal[AgentNode.EXECUTION_OBSERVER, AgentNode.CODE_GENERATION]:
    """
    Route after CODE_EXECUTION_NODE.

    Routes to:
    - EXECUTION_OBSERVER: On success or max retries reached (to generate observations)
    - CODE_GENERATION: On failure with retries remaining

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    signal = state.get("action_signal", None)
    code_generation_attempts = state.get("code_generation_attempts", 0)

    if signal == ActionSignal.CODE_EXECUTION_SUCCESS:
        logger.info("Code executed successfully, routing to EXECUTION_OBSERVER_NODE")
        return AgentNode.EXECUTION_OBSERVER
    elif (
        signal == ActionSignal.CODE_EXECUTION_FAILED
        and code_generation_attempts >= settings.CODE_GENERATION_MAX_RETRIES
    ):
        logger.info(
            f"Code executed with failure and max attempts reached ({code_generation_attempts}), routing to EXECUTION_OBSERVER_NODE"
        )
        return AgentNode.EXECUTION_OBSERVER

    # On failure with attempts left, retry code generation
    logger.info("Code execution failed, routing to CODE_GENERATION_NODE for retry")
    return AgentNode.CODE_GENERATION


def route_after_execution_observer(
    state: AgentState,
) -> Literal[AgentNode.REFLECTION, AgentNode.CODE_PLANNING]:
    """
    Route after EXECUTION_OBSERVER_NODE.

    Routes to:
    - REFLECTION: If execution was successful (to refine and deduplicate observations)
    - CODE_PLANNING: If execution failed (skip reflection, go directly to planning with failure context)

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    current_step_success = state.get("current_step_success", True)

    if not current_step_success:
        logger.info(
            "Execution failed, routing directly to CODE_PLANNING_NODE (skipping reflection)"
        )
        return AgentNode.CODE_PLANNING

    logger.info("Execution successful, routing to REFLECTION_NODE")
    return AgentNode.REFLECTION


def route_after_reflection(
    state: AgentState,
) -> Literal[AgentNode.CODE_PLANNING]:
    """
    Route after REFLECTION_NODE.

    Always routes to CODE_PLANNING to decide next action.
    The LLM in code_planning will decide whether to:
    - Retry the step (on failure)
    - Proceed to next step (on success)
    - Finalize (task complete or giving up)

    Args:
        state: Current agent state

    Returns:
        Next node name
    """
    logger.info("Routing to CODE_PLANNING_NODE after reflection")
    return AgentNode.CODE_PLANNING
