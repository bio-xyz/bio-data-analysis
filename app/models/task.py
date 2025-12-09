from datetime import datetime
from enum import Enum
from typing import Optional

from e2b_code_interpreter import Execution
from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class CompletedStep(BaseModel):
    """Represents a completed execution step in the agent workflow."""

    step_number: int = Field(
        ...,
        description="Sequential step number (1-indexed)",
    )
    goal: str = Field(
        ...,
        description="The goal or objective of this step",
    )
    description: str = Field(
        ...,
        description="Detailed description of what this step does",
    )
    code: str = Field(
        ...,
        description="The Python code executed in this step",
    )
    execution_result: Optional[Execution] = Field(
        None,
        description="Result from code execution (e2b Execution object)",
    )
    success: bool = Field(
        ...,
        description="Whether the step executed successfully",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


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
    base_path: str = Field(
        "",
        description="Optional base path of files in the provided file paths",
    )
    file_paths: list[str] = Field(
        [],
        description="Full paths to the files to be used in the task",
    )
    target_path: Optional[str] = Field(
        None,
        description="Optional target path where the agent should save output files",
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
        base_path: str = Form(""),
        file_paths: list[str] = Form(default=None),
        target_path: Optional[str] = Form(default=None),
    ):
        return cls(
            task_description=task_description,
            data_files_description=data_files_description,
            base_path=base_path,
            file_paths=file_paths or [],
            target_path=target_path,
        )


class ArtifactResponse(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier for the artifact if applicable",
    )
    description: str = Field(
        ...,
        description="Description of the artifact generated during task execution",
    )
    type: str = Field(
        ...,
        description="Type of the artifact, e.g 'FILE', 'FOLDER'",
    )
    content: Optional[str] = Field(
        None,
        description="Content (base64) or reference to the artifact generated during task execution",
    )
    name: Optional[str] = Field(
        None,
        description="Name of the artifact generated during task execution",
    )
    path: Optional[str] = Field(
        None,
        description="Path to the artifact file if applicable",
    )


class TaskResponse(BaseModel):
    id: Optional[str] = Field(
        None,
        description="Unique identifier for the task",
    )
    status: Optional[TaskStatus] = Field(
        None,
        description="Current status of the task",
    )
    answer: str = Field(
        "",
        description="The agent's detailed answer in markdown format",
    )
    artifacts: list[ArtifactResponse] = Field(
        [],
        description="List of artifacts generated during task execution, such as images or tables",
    )
    success: bool = Field(
        True,
        description="Flag indicating whether the task execution was successful",
    )


class TaskStatusResponse(BaseModel):
    id: str = Field(
        ...,
        description="Unique identifier for the submitted task",
    )
    status: TaskStatus = Field(
        ...,
        description="Current status of the task",
    )


class TaskInfo:
    """Holds task execution state and metadata."""

    def __init__(self, task_id: str, status: TaskStatus):
        self.task_id = task_id
        self.status = status
        self.response: Optional[TaskResponse] = None
        self.updated_at = datetime.now()
        self.created_at = datetime.now()

    def update_status(
        self, status: TaskStatus, response: Optional[TaskResponse] = None
    ):
        self.status = status
        if response:
            self.response = response
        self.updated_at = datetime.now()
