import os
from typing import Dict, List

from e2b_code_interpreter import Context, EntryInfo, Execution, Sandbox
from fastapi import UploadFile

from app.config import get_logger, settings
from app.utils import SingletonMeta

logger = get_logger(__name__)


class ExecutorService(metaclass=SingletonMeta):
    """Singleton service to manage E2B sandboxes and code execution."""

    def __init__(self):
        self.sandboxes: Dict[str, Sandbox] = {}
        self.contexts: Dict[str, Context] = {}

    def _validate_sandbox_exists(self, sandbox_id: str) -> None:
        """
        Validate that a sandbox exists.

        Args:
            sandbox_id: Unique identifier for the sandbox

        Raises:
            ValueError: If sandbox does not exist
        """
        if sandbox_id not in self.sandboxes:
            logger.error(f"Sandbox with ID '{sandbox_id}' does not exist")
            raise ValueError(f"Sandbox with ID '{sandbox_id}' does not exist")

    def create_sandbox(self) -> str:
        """
        Create a new E2B sandbox with a specific ID.

        Returns:
            The sandbox ID
        """
        logger.info("Creating new sandbox...")
        sandbox = Sandbox.create()
        sandbox_id = sandbox.get_info().sandbox_id

        self.sandboxes[sandbox_id] = sandbox
        self.create_context(sandbox_id)

        logger.info(f"Sandbox created with ID: {sandbox_id}")
        return sandbox_id

    def destroy_sandbox(self, sandbox_id: str):
        """
        Destroy a specific sandbox by ID.

        Args:
            sandbox_id: Unique identifier for the sandbox to destroy
        """
        self._validate_sandbox_exists(sandbox_id)

        logger.info(f"Destroying sandbox: {sandbox_id}")
        sandbox = self.sandboxes.pop(sandbox_id)
        sandbox.kill()
        del sandbox
        logger.info(f"Sandbox {sandbox_id} destroyed successfully")

    def create_context(self, sandbox_id: str):
        """
        Create new code execution context for a specific sandbox.

        Args:
            sandbox_id: Unique identifier for the sandbox
        """
        self._validate_sandbox_exists(sandbox_id)

        logger.info(f"Creating context for sandbox {sandbox_id}")
        sandbox = self.sandboxes[sandbox_id]
        context = sandbox.create_code_context(cwd=settings.DEFAULT_WORKING_DIRECTORY)
        self.contexts[sandbox_id] = context
        logger.info(f"Context created for sandbox {sandbox_id}")

    def execute_code(self, sandbox_id: str, code: str) -> Execution:
        """
        Execute code in a specific sandbox.

        Args:
            sandbox_id: Unique identifier for the sandbox
            code: Code to execute
        Returns:
            Output from the code execution
        """
        self._validate_sandbox_exists(sandbox_id)

        sandbox = self.sandboxes[sandbox_id]
        context = self.contexts[sandbox_id]

        logger.info(f"Executing code in sandbox {sandbox_id}")
        output = sandbox.run_code(code, context=context)
        logger.info(f"Code execution completed in sandbox {sandbox_id}")

        return output

    async def upload_data_files(
        self,
        sandbox_id: str,
        data_files: List[UploadFile],
        target_folder: str = settings.DEFAULT_DATA_DIRECTORY,
    ) -> List[str]:
        """
        Upload data files to a specific sandbox's user_data folder.

        Args:
            sandbox_id: Unique identifier for the sandbox
            data_files: List of UploadFile objects to upload
            target_folder: Target folder in sandbox (default: /home/user/data)

        Returns:
            List of uploaded file paths
        """
        self._validate_sandbox_exists(sandbox_id)

        sandbox = self.sandboxes[sandbox_id]

        logger.info(f"Uploading {len(data_files)} files to sandbox {sandbox_id}")
        # Create the target folder if it doesn't exist
        sandbox.files.make_dir(target_folder)

        uploaded_files = []
        for upload_file in data_files:
            filename = upload_file.filename
            target_path = f"{target_folder}/{filename}"

            # Read file content from UploadFile
            content = await upload_file.read()

            # Upload to sandbox
            sandbox.files.write(target_path, content)
            uploaded_files.append(target_path)
            logger.info(f"Uploaded file: {filename} to {target_path}")

            # Reset file pointer for potential reuse
            await upload_file.seek(0)

        logger.info(f"Successfully uploaded {len(uploaded_files)} files")
        return uploaded_files

    def download_file(self, sandbox_id: str, file_path: str) -> bytearray:
        """
        Download a file from a specific sandbox.

        Args:
            sandbox_id: Unique identifier for the sandbox
            file_path: Path of the file to download
        Returns:
            Content of the downloaded file as bytes
        """
        self._validate_sandbox_exists(sandbox_id)

        sandbox = self.sandboxes[sandbox_id]

        if not sandbox.files.exists(file_path):
            logger.info("Checking default working directory for the file...")
            file_path = os.path.join(settings.DEFAULT_WORKING_DIRECTORY, file_path)
            if not sandbox.files.exists(file_path):
                logger.error(
                    f"File '{file_path}' does not exist in sandbox '{sandbox_id}'"
                )
                raise FileNotFoundError(
                    f"File '{file_path}' does not exist in sandbox '{sandbox_id}'"
                )

        logger.info(f"Downloading file {file_path} from sandbox {sandbox_id}")
        file_content = sandbox.files.read(file_path, format="bytes")
        logger.info(
            f"File {file_path} downloaded successfully from sandbox {sandbox_id}"
        )
        return file_content

    def list_files(
        self, sandbox_id: str, directory: str = settings.DEFAULT_WORKING_DIRECTORY
    ) -> List[EntryInfo]:
        """
        List files in a specific directory of a sandbox.

        Args:
            sandbox_id: Unique identifier for the sandbox
            directory: Directory to list files from (default: /home/user/workspace)
        Returns:
            List of file paths in the specified directory
        """
        self._validate_sandbox_exists(sandbox_id)

        sandbox = self.sandboxes[sandbox_id]

        logger.info(f"Listing files in directory {directory} of sandbox {sandbox_id}")
        files = sandbox.files.list(directory, depth=25)
        logger.info(
            f"Found {len(files)} files in directory {directory} of sandbox {sandbox_id}"
        )
        return files

    def destroy_all_sandboxes(self):
        """Destroy all sandboxes."""
        logger.info(f"Destroying all sandboxes ({len(self.sandboxes)} total)")
        for sandbox in self.sandboxes.values():
            sandbox.kill()
        self.sandboxes.clear()
        self.contexts.clear()
        logger.info("All sandboxes destroyed")
