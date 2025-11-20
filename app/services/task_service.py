import asyncio
import base64
import uuid
from datetime import datetime
from typing import Optional

from langgraph.graph.state import CompiledStateGraph

from app.config import get_logger, settings
from app.models.task import TaskInfo, TaskRequest, TaskResponse, TaskStatus
from app.services.agent import AgentGraph, AgentState
from app.services.executor_service import ExecutorService
from app.utils import SingletonMeta
from app.utils.datafile import DataFile
from app.utils.nb_builder import NotebookBuilder

logger = get_logger(__name__)


class TaskService(metaclass=SingletonMeta):
    """Singleton service to handle agent operations using LangGraph."""

    def __init__(self):
        self.executor_service: ExecutorService = ExecutorService()
        self.graph: CompiledStateGraph = AgentGraph().get_graph()
        self._tasks: dict[str, TaskInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    def process_task(
        self,
        task: TaskRequest,
        data_files: list[DataFile],
        task_id: str,
    ) -> TaskResponse:
        """
        Process the given task with optional data files using LangGraph workflow.
        Processing works as follows:
            1. Create a sandbox environment for code execution.
            2. Upload provided data files to the sandbox.
            3. Initialize the agent state with task details and uploaded files.
            4. Execute the LangGraph agent workflow.
            5. Collect and return the final task response including any artifacts.

        Args:
            task (TaskRequest): The task request containing the task description and data files description.
            data_files (list[DataFile]): List of data files to be uploaded to the sandbox
            task_id (str): Task ID for tracking and status updates.

        Returns:
            TaskResponse: Result of the task processing
        """

        if not task_id:
            raise ValueError("Task ID must be provided for processing")

        logger.info(f"Processing task: {task.task_description[:50]}...")
        logger.info(f"Number of data files: {len(data_files)}")

        sandbox_id = self.executor_service.create_sandbox()

        try:
            uploaded_files = self.executor_service.upload_data_files(
                sandbox_id, data_files
            )

            # Initialize the agent state
            initial_state = AgentState(
                task_description=task.task_description,
                data_files_description=task.data_files_description,
                uploaded_files=uploaded_files,
                sandbox_id=sandbox_id,
                max_retries=3,
                notebook_builder=NotebookBuilder(),
            )

            logger.info("Starting LangGraph execution...")

            # Execute the graph
            final_state: AgentState = self.graph.invoke(initial_state)

            logger.info("LangGraph execution completed")
            logger.debug(f"Final state: {final_state}")

            # Extract task response from final state
            task_response = final_state["task_response"]

            # Download artifact contents
            for artifact in task_response.artifacts:
                if artifact.path:
                    artifact.content = base64.b64encode(
                        self.executor_service.download_file(sandbox_id, artifact.path)
                    ).decode("utf-8")

        finally:
            self.executor_service.destroy_sandbox(sandbox_id)

        logger.info(f"Task processing completed for sandbox {sandbox_id}")

        task_response.id = task_id
        task_response.status = TaskStatus.COMPLETED
        self.update_task_status(task_id, TaskStatus.COMPLETED, task_response)

        return task_response

    def create_task(self) -> str:
        """
        Create a new task with a unique ID.

        Returns:
            str: The unique task ID
        """
        task_id = str(uuid.uuid4())
        task_info = TaskInfo(task_id, TaskStatus.IN_PROGRESS)
        self._tasks[task_id] = task_info

        # Start cleanup task if not already running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_tasks())

        logger.info(f"Created task with ID: {task_id}")
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        Retrieve a task by ID.

        Args:
            task_id: The unique task ID

        Returns:
            TaskInfo: The task information if found, None otherwise
        """
        return self._tasks.get(task_id)

    def update_task_status(
        self, task_id: str, status: TaskStatus, response: Optional[TaskResponse] = None
    ):
        """
        Update the status and optional response for a task.
        Also updates the updated_at timestamp.

        Args:
            task_id: The unique task ID
            status: The new task status
            response: Optional task response
        """
        task_info = self._tasks.get(task_id)
        if task_info:
            task_info.status = status
            task_info.updated_at = datetime.now()
            if response:
                task_info.response = response
            logger.info(f"Updated task {task_id} status to {status}")

    def process_task_sync(
        self, task: TaskRequest, data_files: list[DataFile]
    ) -> TaskResponse:
        """
        Process task synchronously with automatic task creation and error handling.

        Args:
            task: The task request
            data_files: List of data files to be uploaded

        Returns:
            TaskResponse: The task response with status updated
        """
        task_id = self.create_task()

        try:
            return self.process_task(task, data_files, task_id)
        except Exception as e:
            logger.error(f"Sync task {task_id} failed: {str(e)}", exc_info=True)
            self.update_task_status(
                task_id, TaskStatus.FAILED, self.build_error_response(task_id)
            )
            raise

    def process_task_async(
        self, task_id: str, task: TaskRequest, data_files: list[DataFile]
    ):
        """
        Process task asynchronously in the background.

        Args:
            task_id: The unique task ID
            task: The task request
            data_files: List of data files to be uploaded
        """
        try:
            logger.info(f"Starting async task processing for {task_id}")
            self.process_task(task, data_files, task_id)
            logger.info(f"Async task {task_id} completed successfully")
        except Exception as e:
            logger.error(f"Async task {task_id} failed: {str(e)}", exc_info=True)
            self.update_task_status(
                task_id, TaskStatus.FAILED, self.build_error_response(task_id)
            )

    def build_error_response(self, task_id: Optional[str] = None) -> TaskResponse:
        """
        Build a standardized error response for a failed task.

        Args:
            error: The exception that occurred
        Returns:
            TaskResponse: The standardized error response
        """
        return TaskResponse(
            id=task_id,
            status=TaskStatus.FAILED,
            success=False,
            answer={
                "summary": "Task processing failed",
                "details": [
                    "Please check your task description and data files, then try again.",
                ],
            },
        )

    async def _cleanup_expired_tasks(self):
        """
        Background task to cleanup expired tasks.
        Runs periodically and removes tasks that haven't been accessed recently.
        """
        while True:
            try:
                await asyncio.sleep(settings.TASK_CLEANUP_INTERVAL_SECONDS)

                now = datetime.now()
                expired_task_ids = []

                for task_id, task_info in self._tasks.items():
                    time_since_update = (now - task_info.updated_at).total_seconds()
                    if time_since_update > settings.TASK_EXPIRY_SECONDS:
                        expired_task_ids.append(task_id)

                for task_id in expired_task_ids:
                    logger.info(
                        f"Removing expired task {task_id} "
                        f"(last updated: {self._tasks[task_id].updated_at})"
                    )
                    del self._tasks[task_id]

                if expired_task_ids:
                    logger.info(f"Cleaned up {len(expired_task_ids)} expired tasks")

            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}", exc_info=True)
