"""Agent graph definition using LangGraph.

This module creates the FST-based multi-stage agent graph with:
- PLANNING_NODE: Entry point, decides CODE_PLANNING or ANSWERING
- CODE_PLANNING_NODE: Step-by-step planning and management
- CODE_GENERATION_NODE: Generates code for current step
- CODE_EXECUTION_NODE: Executes code in sandbox
- ANSWERING_NODE: Generates final response

Graph Flow:
    START -> planning -> [code_planning | answering]
    code_planning -> [code_generation | answering]
    code_generation -> code_execution
    code_execution -> [code_planning | code_generation]
    answering -> END
"""

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    answering_node,
    code_execution_node,
    code_generation_node,
    code_planning_node,
    planning_node,
)
from app.agent.signals import AgentNode
from app.agent.state import AgentState
from app.agent.transitions import (
    route_after_code_execution,
    route_after_code_generation,
    route_after_code_planning,
    route_after_planning,
)
from app.config import get_logger
from app.utils import SingletonMeta

logger = get_logger(__name__)


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

        # Add nodes
        graph.add_node(AgentNode.PLANNING, planning_node)
        graph.add_node(AgentNode.CODE_PLANNING, code_planning_node)
        graph.add_node(AgentNode.CODE_GENERATION, code_generation_node)
        graph.add_node(AgentNode.CODE_EXECUTION, code_execution_node)
        graph.add_node(AgentNode.ANSWERING, answering_node)

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

        # code_execution -> code_planning (always goes back to planning)
        graph.add_conditional_edges(
            AgentNode.CODE_EXECUTION,
            route_after_code_execution,
            {
                AgentNode.CODE_GENERATION: AgentNode.CODE_GENERATION,
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
