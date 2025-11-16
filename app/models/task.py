from fastapi import Form
from pydantic import BaseModel, Field, field_validator


class PlanStep(BaseModel):
    step_number: int = Field(
        ...,
        description="The sequential number of the step in the plan",
    )
    title: str = Field(
        ...,
        description="Brief title describing what this step accomplishes",
    )
    description: str = Field(
        ...,
        description="Detailed description of what needs to be done in this step",
    )
    expected_output: str = Field(
        "",
        description="What should be produced or achieved in this step",
    )


class Plan(BaseModel):
    goal: str = Field(
        ...,
        description="Brief description of the overall goal",
    )
    available_resources: list[str] = Field(
        [],
        description="List of available data files and their descriptions",
    )
    steps: list[PlanStep] = Field(
        [],
        description="Sequential steps to accomplish the task",
    )
    expected_artifacts: list[str] = Field(
        [],
        description="List of expected outputs like plots, tables, reports, etc.",
    )


class TaskRequest(BaseModel):
    task_description: str = Field(
        ...,
        description="A detailed description of the task the agent should perform",
        min_length=1,
    )
    data_files_description: str = Field(
        "",
        description="Optional description of the provided data files",
    )

    @field_validator("task_description")
    @classmethod
    def validate_task_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("task_description cannot be empty or whitespace")
        return v.strip()

    @classmethod
    def as_form(
        cls,
        task_description: str = Form(...),
        data_files_description: str = Form(""),
    ):
        return cls(
            task_description=task_description,
            data_files_description=data_files_description,
        )


class AnswerResponse(BaseModel):
    summary: str = Field(
        "",
        description="A concise summary of the agent's findings or results",
    )
    details: list[str] = Field(
        [],
        description="Detailed explanations or observations made by the agent",
    )


class ArtifactResponse(BaseModel):
    description: str = Field(
        "",
        description="Description of the artifact generated during task execution",
    )
    type: str = Field(
        "",
        description="Type of the artifact, e.g., 'image', 'table', 'text', 'csv', 'json', 'plot', 'chart'",
    )
    content: str = Field(
        "",
        description="Content (base64) or reference to the artifact generated during task execution",
    )
    filename: str | None = Field(
        None,
        description="Filename of the artifact generated during task execution",
    )
    path: str | None = Field(
        None,
        description="Path to the artifact file if applicable",
    )
    id: str | None = Field(
        None,
        description="Unique identifier for the artifact if applicable",
    )


class TaskResponse(BaseModel):
    plan: Plan | None = Field(
        None,
        description="The step-by-step plan created to accomplish the task",
    )
    answer: AnswerResponse = Field(
        {},
        description="The agent's answer to the task, including summary and details",
    )
    artifacts: list[ArtifactResponse] = Field(
        [],
        description="List of artifacts generated during task execution, such as images or tables",
    )
