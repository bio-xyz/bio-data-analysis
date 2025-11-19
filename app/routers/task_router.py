from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.models.task import TaskRequest, TaskResponse
from app.services.task_service import TaskService

router = APIRouter()
task_service = TaskService()


@router.post(
    "/task/run", summary="Run agent with code interpreter", response_model=TaskResponse
)
async def run_agent_with_code_interpreter(
    task: Annotated[TaskRequest, Depends(TaskRequest.as_form)],
    data_files: Annotated[list[UploadFile], File(...)] = [],
) -> TaskResponse:
    """
    Run an agent with code interpreter capabilities based on the provided task description and optional data files.

    Args:
        task (TaskRequest): The task request containing the task description and data files description.
        data_files (list[UploadFile]): List of data files to be uploaded to the sandbox.

    Returns:
        dict: The result of the agent's execution.
    """

    return await task_service.process_task(task, data_files)
