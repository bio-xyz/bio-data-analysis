"""Structured output models for instructor-based LLM responses."""

from typing import Literal

from pydantic import BaseModel, Field


class StepObservation(BaseModel):
    """
    Model for capturing important observations from a completed step.

    These observations help track what was learned during code execution
    and inform decisions about next steps.
    """

    title: str = Field(
        ...,
        description="Concise title summarizing the observation (e.g., 'Strong correlation found', 'Missing data in column X')",
    )
    summary: str = Field(
        ...,
        description="Detailed description of what was observed, including specific values, patterns, or findings",
    )
    raw_output: str = Field(
        default="",
        description=(
            "The exact value or content that answers the user's question. "
            "Use when user asks for specific values, exact content, or specifies output format. "
            "Leave empty if summary adequately captures the finding. "
            "Include ONLY the direct answer - exclude everything else from stdout. "
            "Rule: If removing it would make the answer incomplete, keep it; otherwise, exclude it."
        ),
    )
    importance: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Intrinsic strength of the finding (1-5). How strong, reliable, and meaningful is this observation by itself?\n"
            "5 = Critical: Core property, major driver, decisive result\n"
            "4 = Strong: Large effect, clear separation, robust trend\n"
            "3 = Moderate: Clear finding but not dominant\n"
            "2 = Weak: Small effect, noisy result, narrow scope\n"
            "1 = Trivial: Minor detail, sanity check, expected result"
        ),
    )
    relevance: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Usefulness for answering the original task (1-5). How directly does this help answer what the user asked?\n"
            "5 = Essential: Required to answer the question\n"
            "4 = High: Directly informs the question\n"
            "3 = Medium: Some link, helpful background\n"
            "2 = Low: Indirect or contextual only\n"
            "1 = Irrelevant: Interesting but unrelated to the question"
        ),
    )


class PythonCode(BaseModel):
    """
    Model for pure Python code output.

    The code field contains clean, executable Python code without
    markdown fences, backticks, or any other formatting.
    """

    code: str = Field(
        ...,
        description=(
            "CRITICAL: Pure executable Python code. Must NOT contain markdown code fences "
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
    step_goal: str = Field(
        default="",
        description="Clear, small, specific goal for the current/next step (empty string if TASK_COMPLETED or TASK_FAILED)",
    )
    step_description: str = Field(
        default="",
        description="Detailed description of what needs to be done (empty string if TASK_COMPLETED or TASK_FAILED) in markdown format",
    )
    reasoning: str = Field(
        default="",
        description="Explanation of why this decision was made",
    )
    observations: list[StepObservation] = Field(
        default_factory=list,
        description=(
            "List of important observations from the current step. "
            "Populate when signal is PROCEED_TO_NEXT_STEP, TASK_COMPLETED, or TASK_FAILED. "
            "Capture key findings, patterns, statistics, or issues discovered during execution. "
            "Each observation should have meaningful importance and relevance scores. "
            "Return empty list if no useful data insights were found."
        ),
    )


class ArtifactDecision(BaseModel):
    """Model for artifact selection decision."""

    type: Literal["FOLDER", "FILE"] = Field(
        ...,
        description="Type of artifact: FOLDER or FILE",
    )
    description: str = Field(
        ...,
        description="Description of the generated artifact",
    )
    full_path: str = Field(
        ...,
        description="Full path to the generated artifact. For FOLDER, provide the full folder path; for FILE, provide full path to the file",
    )


class TaskResponseAnswer(BaseModel):
    """Model for task response output."""

    notebook_description: str = Field(
        ...,
        description=(
            "Title/description for the Jupyter notebook based on the analysis. "
            "1 sentence, max 5-10 words. Must start with a capital letter, ending with period if possible. "
            "Be specific to the task (mention main goal, data, or method). "
            "Should include 'Jupyter notebook' or 'analysis notebook' or similar phrase. "
            "Output ONLY the sentence, no quotes, no extra text."
        ),
    )
    answer: str = Field(
        ...,
        description=(
            "The answer to the user's task. FORMAT DEPENDS ON USER REQUEST:\n\n"
            "**IF user specified output format** (e.g., 'answer must be just a number', 'respond with only yes/no', "
            "'output as CSV', 'answer with Not Applicable if...'):\n"
            "  - Return ONLY the exact answer in that format\n"
            "  - Use raw_output from observations if available\n"
            "  - Examples: '42', 'Not Applicable', 'yes', 'value1,value2,value3'\n\n"
            "**OTHERWISE**, provide a Markdown report with this structure:\n\n"
            "# [Task-Specific Title]\n"
            "2-5 sentence summary of the main conclusion or failure explanation.\n\n"
            "## Key Findings\n"
            "3-10 bullet points with concrete numbers/facts.\n\n"
            "## Results and Interpretation\n"
            "Grouped explanations with specific values. Reference artifacts as [FILENAME].\n\n"
            "## Limitations\n"
            "Data quality issues, missing data, or analysis constraints.\n\n"
            "## Generated Artifacts\n"
            "List of selected artifacts with descriptions.\n\n"
            "## Conclusions and Implications\n"
            "What the findings mean for the user's question."
        ),
    )
    success: bool = Field(
        default=True,
        description=(
            "Whether the task was completed successfully.\n"
            "- True: Task completed and produced meaningful results (even if partial).\n"
            "- False: Task failed due to errors, missing data, or inability to answer."
        ),
    )
    artifacts: list[ArtifactDecision] = Field(
        default_factory=list,
        description=(
            "List of artifacts to return to the user.\n"
            "- Select FILE for individual plots, tables, CSVs.\n"
            "- Select FOLDER only for large datasets (>20 files) or logical units.\n"
            "- Empty list if no artifacts were generated or relevant."
        ),
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
