"""Structured output models for instructor-based LLM responses."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PythonCode(BaseModel):
    """
    Model for pure Python code output.

    The code field contains clean, executable Python code without
    markdown fences, backticks, or any other formatting.
    """

    code: str = Field(
        ...,
        description=(
            "Pure executable Python code. Must NOT contain markdown code fences "
            "(```python or ```), backticks, or any other formatting. "
            "Should be directly executable as-is."
        ),
    )


class PlanningDecision(BaseModel):
    """Model for planning node decisions."""

    signal: Literal["CODE_PLANNING", "GENERAL_ANSWER", "CLARIFICATION"] = Field(
        ...,
        description="The decision signal: CODE_PLANNING, GENERAL_ANSWER, or CLARIFICATION",
    )
    rationale: str = Field(
        ...,
        description="Detailed explanation of the task for CODE_PLANNING, OR reason for choosing GENERAL_ANSWER OR reason for choosing CLARIFICATION.",
    )


class CodePlanningDecision(BaseModel):
    """Model for code planning node decisions."""

    signal: Literal[
        "ITERATE_CURRENT_STEP", "PROCEED_TO_NEXT_STEP", "TASK_COMPLETED", "TASK_FAILED"
    ] = Field(
        ...,
        description="The decision signal: ITERATE_CURRENT_STEP, PROCEED_TO_NEXT_STEP, TASK_COMPLETED, or TASK_FAILED",
    )
    current_step_goal: str = Field(
        default="",
        description="Clear, small, specific goal for the current/next step (empty string if TASK_COMPLETED or TASK_FAILED)",
    )
    current_step_description: str = Field(
        default="",
        description="Detailed description of what needs to be done (empty string if TASK_COMPLETED or TASK_FAILED) in markdown format",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why this decision was made",
    )
    progress_summary: str = Field(
        default="",
        description="Brief summary of overall progress so far",
    )


class ArtifactInfo(BaseModel):
    """Model for artifact information."""

    id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the artifact if not saved on disk, e.g., in-memory artifact",
    )
    type: str = Field(
        default="unknown",
        description="Type of artifact (png|chart|table|csv|json|text|plot)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the saved or execution result artifact",
    )
    filename: str = Field(default=None, description="Artifact filename")
    path: Optional[str] = Field(
        default=None, description="Path to the artifact if applicable and saved on disk"
    )


class TaskResponseOutput(BaseModel):
    """Model for task response output."""

    answer: str = Field(
        ...,
        description="Direct answer to the user's question if task did not require code execution. Or detailed report if code execution was performed.\n"
        "Example format for report:\n\n"
        "# Task-Specific Title\n\n## Overview\nBrief summary...\n\n## Key Findings\n- IC50: 6.236 µM\n- R²: 0.9976\n\n## Results and Interpretation\nDetailed analysis with inline artifact references [dose_response_curve.png]...\n\n**Model Fit Parameters:**\n- Parameter a: value\n- Parameter b: value\n\n## Data Patterns and Insights\nObservations...\n\n**Summary statistics:**\n- Median: value\n- Range: min - max\n\n## Generated Artifacts\n- **dose_response_curve.png**: Description\n- **residuals_plot.png**: Description\n\n## Conclusions\nMain takeaways...",
    )
    success: bool = Field(
        default=True,
        description="Whether the task was completed successfully",
    )
    artifacts: list[ArtifactInfo] = Field(
        default_factory=list,
        description="List of artifacts generated during the task. Can be empty if no artifacts were created or if user did not request them.",
    )


class ClarificationResponse(BaseModel):
    """Model for clarification questions response."""

    questions: str = Field(
        ...,
        description=(
            "Clarification questions to ask the user. Should be formatted as "
            "a clear, helpful message explaining what additional information is needed."
        ),
    )


class GeneralAnswerResponse(BaseModel):
    """Model for general answer response (no code execution needed)."""

    answer: str = Field(
        ...,
        description=(
            "A comprehensive answer to the user's question. Should be well-formatted "
            "markdown that directly addresses the user's request without requiring "
            "code execution."
        ),
    )
