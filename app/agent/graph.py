"""Agent graph definition using LangGraph.

This module creates the FST-based multi-stage agent graph with:
- PLANNING_NODE: Entry point, decides CODE_PLANNING or ANSWERING
- CODE_PLANNING_NODE: Step-by-step planning and management
- CODE_GENERATION_NODE: Generates code for current step
- CODE_EXECUTION_NODE: Executes code in sandbox
- EXECUTION_OBSERVER_NODE: Generates observations from execution results
- REFLECTION_NODE: Refines and deduplicates world observations
- ANSWERING_NODE: Generates final response

Graph Flow:
    START -> planning -> [code_planning | answering]
    code_planning -> [code_generation | answering]
    code_generation -> code_execution
    code_execution -> [execution_observer | code_generation]
    execution_observer -> [reflection | code_planning]
    reflection -> code_planning
    answering -> END
"""

from functools import wraps
from typing import Callable

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    answering_node,
    code_execution_node,
    code_generation_node,
    code_planning_node,
    execution_observer_node,
    planning_node,
    reflection_node,
)
from app.agent.signals import AgentNode
from app.agent.state import AgentState
from app.agent.transitions import (
    route_after_code_execution,
    route_after_code_generation,
    route_after_code_planning,
    route_after_execution_observer,
    route_after_planning,
    route_after_reflection,
)
from app.config import get_logger
from app.models.task import TaskStatus
from app.utils import SingletonMeta

logger = get_logger(__name__)


def with_status_update(
    node_func: Callable[[AgentState], dict],
) -> Callable[[AgentState], dict]:
    """
    Decorator that updates task status to IN_PROGRESS before each node execution.

    Args:
        node_func: The node function to wrap

    Returns:
        Wrapped function that updates status before execution
    """

    @wraps(node_func)
    def wrapper(state: AgentState) -> dict:
        task_info = state.get("task_info")
        if task_info:
            task_info.update_status(TaskStatus.IN_PROGRESS)
        return node_func(state)

    return wrapper


class AgentGraph(metaclass=SingletonMeta):
    """Singleton for the compiled agent graph."""

    def __init__(self):
        """Initialize and compile the agent graph."""
        logger.info("Creating agent graph...")
        self.graph = self._create_graph()
        logger.info("Agent graph created successfully")

    def _create_graph(self) -> StateGraph:
        """
        Create the agent graph with FST-based multi-stage architecture.

        Returns:
            Compiled StateGraph
        """
        # Create graph with AgentState
        graph = StateGraph(AgentState)

        # Add nodes (wrapped with status update)
        graph.add_node(AgentNode.PLANNING, with_status_update(planning_node))
        graph.add_node(AgentNode.CODE_PLANNING, with_status_update(code_planning_node))
        graph.add_node(
            AgentNode.CODE_GENERATION, with_status_update(code_generation_node)
        )
        graph.add_node(
            AgentNode.CODE_EXECUTION, with_status_update(code_execution_node)
        )
        graph.add_node(
            AgentNode.EXECUTION_OBSERVER, with_status_update(execution_observer_node)
        )
        graph.add_node(AgentNode.REFLECTION, with_status_update(reflection_node))
        graph.add_node(AgentNode.ANSWERING, with_status_update(answering_node))

        # Add edges
        # START -> planning
        graph.add_edge(START, AgentNode.PLANNING)

        # planning -> code_planning | answering
        graph.add_conditional_edges(
            AgentNode.PLANNING,
            route_after_planning,
            {
                AgentNode.CODE_PLANNING: AgentNode.CODE_PLANNING,
                AgentNode.ANSWERING: AgentNode.ANSWERING,
            },
        )

        # code_planning -> code_generation | answering
        graph.add_conditional_edges(
            AgentNode.CODE_PLANNING,
            route_after_code_planning,
            {
                AgentNode.CODE_GENERATION: AgentNode.CODE_GENERATION,
                AgentNode.ANSWERING: AgentNode.ANSWERING,
            },
        )

        # code_generation -> code_execution
        graph.add_conditional_edges(
            AgentNode.CODE_GENERATION,
            route_after_code_generation,
            {
                AgentNode.CODE_EXECUTION: AgentNode.CODE_EXECUTION,
            },
        )

        # code_execution -> execution_observer | code_generation
        graph.add_conditional_edges(
            AgentNode.CODE_EXECUTION,
            route_after_code_execution,
            {
                AgentNode.EXECUTION_OBSERVER: AgentNode.EXECUTION_OBSERVER,
                AgentNode.CODE_GENERATION: AgentNode.CODE_GENERATION,
            },
        )

        # execution_observer -> reflection | code_planning
        graph.add_conditional_edges(
            AgentNode.EXECUTION_OBSERVER,
            route_after_execution_observer,
            {
                AgentNode.REFLECTION: AgentNode.REFLECTION,
                AgentNode.CODE_PLANNING: AgentNode.CODE_PLANNING,
            },
        )

        # reflection -> code_planning
        graph.add_conditional_edges(
            AgentNode.REFLECTION,
            route_after_reflection,
            {
                AgentNode.CODE_PLANNING: AgentNode.CODE_PLANNING,
            },
        )

        # answering -> END
        graph.add_edge(AgentNode.ANSWERING, END)

        # Compile the graph
        compiled = graph.compile()
        logger.info("Graph compiled successfully")

        return compiled

    def get_graph(self) -> StateGraph:
        """
        Get the compiled agent graph.

        Returns:
            Compiled StateGraph instance
        """
        return self.graph
