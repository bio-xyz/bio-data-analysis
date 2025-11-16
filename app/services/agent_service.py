import base64

from fastapi import UploadFile

from app.config import get_logger, settings
from app.models.task import TaskRequest, TaskResponse
from app.services.executor_service import ExecutorService
from app.services.llm import LLMService

logger = get_logger(__name__)


class AgentService:
    """Service to handle agent operations."""

    def __init__(self):
        self.executor_service = ExecutorService()
        self.llm_planner = LLMService(llm_config=settings.PLAN_GENERATION_LLM)
        self.llm_code_generator = LLMService(llm_config=settings.CODE_GENERATION_LLM)
        self.llm_response_generator = LLMService(
            llm_config=settings.RESPONSE_GENERATION_LLM
        )

    async def process_task(
        self, task: TaskRequest, data_files: list[UploadFile]
    ) -> TaskResponse:
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

        try:
            uploaded_files = await self.executor_service.upload_data_files(
                sandbox_id, data_files
            )

            # Step 1: Generate a plan
            logger.info("Step 1: Generating plan...")
            plan = self.llm_planner.generate_plan(
                task_description=task.task_description,
                data_files_description=task.data_files_description,
                uploaded_files=uploaded_files,
            )
            logger.info(f"Plan generated with {len(plan.steps)} steps")
            logger.debug(f"Plan: {plan}")

            # Step 2: Generate code based on the plan
            logger.info("Step 2: Generating code based on plan...")
            generated_code = self.llm_code_generator.generate_code(
                task_description=task.task_description,
                data_files_description=task.data_files_description,
                uploaded_files=uploaded_files,
                plan=plan,
            )
            logger.debug(f"Generated code: {generated_code}")

            # Step 3: Execute the code
            logger.info("Step 3: Executing generated code in sandbox...")
            execution_result = self.executor_service.execute_code(
                sandbox_id, generated_code
            )
            logger.debug(f"Execution result: {execution_result}")

            # Step 4: Generate response
            logger.info("Step 4: Generating task response...")
            task_response = self.llm_response_generator.generate_task_response(
                task_description=task.task_description,
                generated_code=generated_code,
                execution_result=execution_result,
            )

            # Add the plan to the response
            task_response.plan = plan

            for artifact in task_response.artifacts:
                if artifact.path:
                    artifact.content = base64.b64encode(
                        self.executor_service.download_file(sandbox_id, artifact.path)
                    ).decode("utf-8")
        finally:
            self.executor_service.destroy_sandbox(sandbox_id)

        logger.info(f"Task processing completed for sandbox {sandbox_id}")

        return task_response
