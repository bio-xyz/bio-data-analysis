from fastapi import UploadFile

from app.config import get_logger
from app.models.task import TaskRequest
from app.services.executor_service import ExecutorService

logger = get_logger(__name__)


class AgentService:
    """Service to handle agent operations."""

    def __init__(self):
        self.executor_service = ExecutorService()

    async def process_task(self, task: TaskRequest, data_files: list[UploadFile]):
        """
        Process the given task with optional data files. Agent generates the code, executes in a sandbox, and returns results.

        Args:
            task (TaskRequest): The task request containing the task description and data files description.
            data_files (list[UploadFile]): List of data files to be uploaded to the sandbox

        Returns:
            dict: Result of the task processing
        """

        logger.info(f"Processing task: {task.task_description[:50]}...")
        logger.info(f"Number of data files: {len(data_files)}")

        sandbox_id = self.executor_service.create_sandbox()

        uploaded_files = await self.executor_service.upload_data_files(
            sandbox_id, data_files
        )

        self.executor_service.destroy_sandbox(sandbox_id)

        logger.info(f"Task processing completed for sandbox {sandbox_id}")
        return {
            "filenames": [file.filename for file in data_files],
            "task": task.task_description,
            "data_files_description": task.data_files_description,
            "uploaded_files": uploaded_files,
            "sandbox_id": sandbox_id,
        }
