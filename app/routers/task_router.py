from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from app.config import get_logger
from app.models.task import TaskRequest, TaskResponse, TaskStatus, TaskStatusResponse
from app.services.task_service import TaskService
from app.utils.datafile import convert_upload_files_to_data_files

logger = get_logger(__name__)

router = APIRouter()
task_service = TaskService()


@router.post(
    "/task/run/sync",
    summary="Run agent with code interpreter",
    response_model=TaskResponse,
    responses={status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": TaskResponse}},
    response_model_exclude_none=True,
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
        validated_data_files = await convert_upload_files_to_data_files(data_files)
        return task_service.process_task_sync(task, validated_data_files)
    except Exception as e:
        logger.error(f"Task processing failed: {str(e)}", exc_info=True)
        error_response = task_service.build_error_response()
        return JSONResponse(
            content=error_response.model_dump(),
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )


@router.post(
    "/task/run/async",
    summary="Run agent with code interpreter asynchronously",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    response_model_exclude_none=True,
)
async def run_agent_async(
    background_tasks: BackgroundTasks,
    task: Annotated[TaskRequest, Depends(TaskRequest.as_form)],
    data_files: Annotated[list[UploadFile], File(...)] = [],
) -> TaskStatusResponse:
    """
    Submit a task to run asynchronously in the background.

    Args:
        background_tasks (BackgroundTasks): FastAPI BackgroundTasks for async processing.
        task (TaskRequest): The task request containing the task description and data files description.
        data_files (list[UploadFile]): List of data files to be uploaded to the sandbox.

    Returns:
        TaskStatusResponse: Contains task_id and initial status (IN_PROGRESS)
    """
    validated_data_files = await convert_upload_files_to_data_files(data_files)

    task_id = task_service.create_task()

    background_tasks.add_task(
        task_service.process_task_async, task_id, task, validated_data_files
    )

    logger.info(f"Task {task_id} submitted for async processing")

    return TaskStatusResponse(id=task_id, status=TaskStatus.IN_PROGRESS)


@router.get(
    "/task/{task_id}",
    summary="Get task status and details",
    response_model=TaskResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Task not found"},
        status.HTTP_200_OK: {"model": TaskResponse},
    },
    response_model_exclude_none=True,
)
async def get_task_details(task_id: str) -> TaskResponse:
    """
    Retrieve the status and details of a task by its ID.

    Args:
        task_id: The unique task ID

    Returns:
        TaskResponse: Complete task response including status and results

    Raises:
        HTTPException: 404 if task not found
    """
    task_info = task_service.get_task(task_id)

    if not task_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    # If task is still in progress, return partial response
    if task_info.status == TaskStatus.IN_PROGRESS:
        return TaskResponse(
            id=task_id,
            status=TaskStatus.IN_PROGRESS,
            success=True,
            answer="Task is still processing. Please check back later for results.",
        )

    # Return the completed or failed response
    if task_info.response:
        return task_info.response

    # Fallback for edge cases
    return TaskResponse(
        id=task_id,
        status=task_info.status,
        success=False,
        answer=f"Task is {task_info.status.value}. No response data available",
    )
