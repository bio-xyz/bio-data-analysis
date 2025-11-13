from typing import Dict, List

from e2b_code_interpreter import Context, Sandbox
from fastapi import UploadFile

from app.config import get_logger

logger = get_logger(__name__)


class ExecutorService:
    def __init__(self):
        self.sandboxes: Dict[str, Sandbox] = {}
        self.contexts: Dict[str, Context] = {}

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
        if sandbox_id not in self.sandboxes:
            logger.error(f"Sandbox with ID '{sandbox_id}' does not exist")
            raise ValueError(f"Sandbox with ID '{sandbox_id}' does not exist")

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
        if sandbox_id not in self.sandboxes:
            logger.error(f"Sandbox with ID '{sandbox_id}' does not exist")
            raise ValueError(f"Sandbox with ID '{sandbox_id}' does not exist")

        logger.info(f"Creating context for sandbox {sandbox_id}")
        sandbox = self.sandboxes[sandbox_id]
        context = sandbox.create_code_context()
        self.contexts[sandbox_id] = context
        logger.info(f"Context created for sandbox {sandbox_id}")

    async def upload_data_files(
        self,
        sandbox_id: str,
        data_files: List[UploadFile],
        target_folder: str = "/home/user/data",
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
        if sandbox_id not in self.sandboxes:
            logger.error(f"Sandbox with ID '{sandbox_id}' does not exist")
            raise ValueError(f"Sandbox with ID '{sandbox_id}' does not exist")

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

    def destroy_all_sandboxes(self):
        """Destroy all sandboxes."""
        logger.info(f"Destroying all sandboxes ({len(self.sandboxes)} total)")
        for sandbox in self.sandboxes.values():
            sandbox.kill()
        self.sandboxes.clear()
        logger.info("All sandboxes destroyed")
