"""Agent state definition for LangGraph."""

from typing import Any

from langgraph.graph import MessagesState
from pydantic import Field

from app.config import settings
from app.models.task import Plan, TaskResponse


class AgentState(MessagesState):
    """
    State for the data science agent.

    Tracks the agent's progress through the workflow including
    task description, plan, code, execution results, and current state.
    """

    # Input data
    task_description: str = Field(default="", description="User's task description")
    data_files_description: str = Field(
        default="", description="Description of uploaded data files"
    )
    uploaded_files: list[str] = Field(
        default_factory=list, description="List of uploaded file names"
    )

    # Planning
    plan: Plan | None = Field(default=None, description="Generated execution plan")

    # Code generation
    generated_code: str = Field(default="", description="Generated Python code")
    code_generation_attempts: int = Field(
        default=0, description="Number of code generation attempts"
    )

    # Execution
    execution_result: Any = Field(
        default=None, description="Result from code execution"
    )
    execution_error: str | None = Field(
        default=None, description="Error message if execution failed"
    )
    has_execution_error: bool = Field(
        default=False, description="Whether execution had errors"
    )

    # Final response
    task_response: TaskResponse = Field(default=None, description="Final task response")

    # Agent control
    action_signal: str = Field(default="continue", description="Signal for next action")
    sandbox_id: str | None = Field(
        default=None, description="Sandbox ID for code execution"
    )

    # Retry logic
    max_retries: int = Field(
        default=settings.CODE_GENERATION_MAX_RETRIES,
        description="Maximum number of retry attempts for code generation",
    )
