"""Agent state definition for LangGraph."""

from typing import Optional

from e2b_code_interpreter import Execution
from langgraph.graph import MessagesState
from pydantic import Field

from app.models.structured_outputs import TaskResponseAnswer
from app.models.task import CompletedStep, TaskInfo


class AgentState(MessagesState):
    """
    State for the data science agent.

    Graph Flow:
    - PLANNING_NODE: Entry point, decides between code execution or direct answer
    - CODE_PLANNING_NODE: Plans and manages step-by-step code execution
    - CODE_GENERATION_NODE: Generates code for current step
    - CODE_EXECUTION_NODE: Executes code in sandbox
    - ANSWERING_NODE: Generates final response

    Tracks the agent's progress through the workflow including
    task description, code, execution results, and current state.
    """

    # Input data
    task_description: str = Field(description="User's task description")
    data_files_description: str = Field(
        default="", description="Description of uploaded data files"
    )
    uploaded_files: list[str] = Field(
        default_factory=list, description="List of uploaded file names"
    )

    # Task execution environment
    sandbox_id: str = Field(description="Sandbox ID for code execution")
    task_info: TaskInfo = Field(description="TaskInfo object for tracking task status")

    # Agent control
    action_signal: str = Field(default="continue", description="Signal for next action")

    # PLANNING_NODE outputs
    task_rationale: str = Field(
        default="", description="Rationale/reasoning about the task from PLANNING_NODE"
    )

    # CODE_PLANNING_NODE state - Step management
    current_step_goal: str = Field(
        default="", description="Current step goal to be executed"
    )
    current_step_description: str = Field(
        default="", description="Detailed description of current step"
    )
    current_step_goal_history: list[str] = Field(
        default_factory=list, description="History of current step goals tried"
    )
    step_number: int = Field(default=0, description="Current step number (1-indexed)")
    step_attempts: int = Field(
        default=0, description="Number of attempts for current step"
    )
    completed_steps: list[CompletedStep] = Field(
        default_factory=list, description="List of completed steps with results"
    )

    # CODE_GENERATION_NODE state
    generated_code: str = Field(
        default="", description="Generated Python code for current step"
    )
    code_generation_attempts: int = Field(
        default=0, description="Total number of code generation attempts"
    )

    # CODE_EXECUTION_NODE state
    execution_result: Optional[Execution] = Field(
        default=None, description="Result from code execution"
    )
    last_execution_output: str = Field(
        default="", description="Output from last code execution"
    )
    last_execution_error: Optional[str] = Field(
        default=None, description="Error from last code execution"
    )

    # Overall status
    error: Optional[str] = Field(
        default=None, description="Error message from any stage"
    )
    failure_reason: str = Field(default="", description="Reason for failure if any")

    # ANSWERING_NODE output
    task_answer: Optional[TaskResponseAnswer] = Field(
        default=None, description="Final task response"
    )
