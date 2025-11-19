import base64

from fastapi import UploadFile
from langgraph.graph.state import CompiledStateGraph

from app.config import get_logger
from app.models.task import TaskRequest, TaskResponse
from app.services.agent import AgentGraph, AgentState
from app.services.executor_service import ExecutorService
from app.utils import SingletonMeta

logger = get_logger(__name__)


class TaskService(metaclass=SingletonMeta):
    """Singleton service to handle agent operations using LangGraph."""

    def __init__(self):
        self.executor_service: ExecutorService = ExecutorService()
        self.graph: CompiledStateGraph = AgentGraph().get_graph()

    async def process_task(
        self, task: TaskRequest, data_files: list[UploadFile]
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
            data_files (list[UploadFile]): List of data files to be uploaded to the sandbox

        Returns:
            TaskResponse: Result of the task processing
        """

        logger.info(f"Processing task: {task.task_description[:50]}...")
        logger.info(f"Number of data files: {len(data_files)}")

        sandbox_id = self.executor_service.create_sandbox()

        try:
            uploaded_files = await self.executor_service.upload_data_files(
                sandbox_id, data_files
            )

            # Initialize the agent state
            initial_state = AgentState(
                task_description=task.task_description,
                data_files_description=task.data_files_description,
                uploaded_files=uploaded_files,
                sandbox_id=sandbox_id,
                max_retries=3,
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

        return task_response
