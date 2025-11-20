from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import get_logger
from app.models.task import AnswerResponse, TaskRequest, TaskResponse
from app.services.task_service import TaskService

logger = get_logger(__name__)

router = APIRouter()
task_service = TaskService()


@router.post(
    "/task/run",
    summary="Run agent with code interpreter",
    response_model=TaskResponse,
    responses={status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": TaskResponse}},
)
async def run_agent_with_code_interpreter(
    task: Annotated[TaskRequest, Depends(TaskRequest.as_form)],
    data_files: Annotated[list[UploadFile], File(...)] = [],
) -> TaskResponse | JSONResponse:
    """
    Run an agent with code interpreter capabilities based on the provided task description and optional data files.

    Args:
        task (TaskRequest): The task request containing the task description and data files description.
        data_files (list[UploadFile]): List of data files to be uploaded to the sandbox.

    Returns:
        TaskResponse: The result of the agent's execution, or an error response with success=False.
    """

    try:
        return await task_service.process_task(task, data_files)
    except Exception as e:
        logger.error(f"Task processing failed: {str(e)}", exc_info=True)

        error_response = TaskResponse(
            success=False,
            answer=AnswerResponse(
                summary="Task processing failed",
                details=[
                    "Please check your task description and data files, then try again.",
                ],
            ),
        )

        return JSONResponse(
            content=error_response.model_dump(),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
