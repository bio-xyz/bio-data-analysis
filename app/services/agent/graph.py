"""Main LangGraph definition for the data science agent.

Graph Flow:

    START
      ↓
    [plan]  ───────────→  Generate execution plan
      ↓
    [code_generation]  ──→  Generate Python code
      ↓
    [execution]  ────────→  Execute code in sandbox
      ↓
      ├─ SUCCESS ────────→  [analyze]  ──→  Generate response  ──→  END
      │
      └─ ERROR ──────────→  [code_generation]  (retry with error context)
                            ↓ (if max_retries reached)
                            [analyze]  ──→  Generate response with error  ──→  END

"""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import get_logger
from app.services.agent.nodes import (
    AgentNode,
    analyze_node,
    code_generation_node,
    execution_node,
    plan_node,
)
from app.services.agent.state import AgentState
from app.services.agent.transitions import should_regenerate_code
from app.utils.singleton import SingletonMeta

logger = get_logger(__name__)


class AgentGraph(metaclass=SingletonMeta):
    """
    Singleton class that manages the agent workflow graph.

    The graph is compiled once and reused across all requests.
    """

    def __init__(self):
        """Initialize and compile the agent workflow graph."""
        logger.info("Creating agent graph...")
        self._graph: CompiledStateGraph = self._create_graph()
        logger.info("Agent graph created successfully")

    def _create_graph(self) -> CompiledStateGraph:
        """
        Create and compile the agent workflow graph.

        Returns:
            Compiled StateGraph ready for execution
        """
        # Create the graph with AgentState
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node(AgentNode.PLAN, plan_node)
        workflow.add_node(AgentNode.CODE_GENERATION, code_generation_node)
        workflow.add_node(AgentNode.EXECUTION, execution_node)
        workflow.add_node(AgentNode.ANALYZE, analyze_node)

        # Set entry point
        workflow.set_entry_point(AgentNode.PLAN)

        # Add edges
        # Plan -> Code Generation (always)
        workflow.add_edge(AgentNode.PLAN, AgentNode.CODE_GENERATION)

        # Code Generation -> Execution (always)
        workflow.add_edge(AgentNode.CODE_GENERATION, AgentNode.EXECUTION)

        # Execution -> Code Generation (if error) OR Analyze (if success)
        workflow.add_conditional_edges(
            AgentNode.EXECUTION,
            should_regenerate_code,
            {
                AgentNode.CODE_GENERATION: AgentNode.CODE_GENERATION,  # Retry on error
                AgentNode.ANALYZE: AgentNode.ANALYZE,  # Success or max retries
            },
        )

        # Analyze -> END
        workflow.add_edge(AgentNode.ANALYZE, END)

        # Compile the graph
        return workflow.compile()

    def get_graph(self) -> CompiledStateGraph:
        """Get the compiled graph instance."""
        return self._graph
