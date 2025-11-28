from datetime import datetime
from enum import Enum

from fastapi import Form
from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


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
    id: str | None = Field(
        None,
        description="Unique identifier for the task",
    )
    status: TaskStatus | None = Field(
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
        self.response: TaskResponse | None = None
        self.updated_at = datetime.now()
        self.created_at = datetime.now()

    def update_status(self, status: TaskStatus, response: TaskResponse | None = None):
        self.status = status
        if response:
            self.response = response
        self.updated_at = datetime.now()
