import os
import shlex
from pathlib import Path
from typing import Dict, List

import nbformat
from e2b_code_interpreter import Context, Execution, Sandbox

from app.config import get_logger, settings
from app.utils import SingletonMeta
from app.utils.datafile import DataFile

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
        sandbox = Sandbox.create(
            template=settings.SANDBOX_TEMPLATE,
            timeout=settings.SANDBOX_DEFAULT_TIMEOUT_SECONDS,
        )
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

    def _run_script_in_sandbox(
        self,
        sandbox_id: str,
        local_script_path: str,
        sandbox_script_path: str,
        command_args: List[str],
        command_prefix: str = "",
        env_vars: Dict[str, str] = None,
    ):
        """
        Helper method to upload a script to sandbox, make it executable, and run it.

        Args:
            sandbox_id: Unique identifier for the sandbox
            local_script_path: Path to the script file on local filesystem
            sandbox_script_path: Path where script should be written in sandbox
            command_args: List of command arguments to pass to the script
            command_prefix: Optional command prefix (e.g., "sudo", "python", "bash", etc.)
            env_vars: Optional dictionary of environment variables

        Returns:
            Command execution result
        """
        self._validate_sandbox_exists(sandbox_id)
        sandbox = self.sandboxes[sandbox_id]

        try:
            with open(local_script_path, "r") as f:
                script_content = f.read()
            sandbox.files.write(sandbox_script_path, script_content)
            sandbox.commands.run(f"sudo chmod +x {shlex.quote(sandbox_script_path)}")

            quoted_args = [shlex.quote(sandbox_script_path)] + [
                shlex.quote(arg) for arg in command_args
            ]
            command = (
                f"{command_prefix} {' '.join(quoted_args)}"
                if command_prefix
                else " ".join(quoted_args)
            )

            result = sandbox.commands.run(
                command,
                envs=env_vars or {},
            )

            return result
        finally:
            if sandbox.files.exists(sandbox_script_path):
                sandbox.files.remove(sandbox_script_path)

    def upload_data_files(
        self,
        sandbox_id: str,
        data_files: List[DataFile],
        target_folder: str = settings.DEFAULT_DATA_DIRECTORY,
    ) -> List[str]:
        """
        Upload data files to a specific sandbox's user_data folder.

        Args:
            sandbox_id: Unique identifier for the sandbox
            data_files: List of DataFile objects to upload
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
        for data_file in data_files:
            filename = data_file.filename
            target_path = f"{target_folder}/{filename}"

            # Upload to sandbox using bytes content from DataFile
            write_info = sandbox.files.write(target_path, data_file.content)
            uploaded_files.append(write_info.path)
            logger.info(
                f"Uploaded file: {filename} to {target_path} (size: {data_file.size} bytes)"
            )

        logger.info(f"Successfully uploaded {len(uploaded_files)} files")
        return uploaded_files

    def save_notebook_to_sandbox(
        self,
        sandbox_id: str,
        notebook: nbformat.NotebookNode,
        notebook_filename: str = settings.DEFAULT_NOTEBOOK_FILENAME,
        notebook_path: str = settings.DEFAULT_WORKING_DIRECTORY,
    ) -> str:
        """
        Save a Jupyter notebook to a specific sandbox.
        Args:
            sandbox_id: Unique identifier for the sandbox
            notebook: Jupyter notebook object to save
            notebook_filename: Filename for the notebook in the sandbox
        Returns:
            Path to the saved notebook in the sandbox
        """
        self._validate_sandbox_exists(sandbox_id)

        sandbox = self.sandboxes[sandbox_id]

        notebook_path = os.path.join(notebook_path, notebook_filename)
        notebook_content = nbformat.writes(notebook)

        sandbox.files.write(notebook_path, notebook_content)

        logger.info(f"Notebook saved to sandbox {sandbox_id} at path {notebook_path}")

        return notebook_path

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

    def download_from_s3(
        self,
        sandbox_id: str,
        s3_paths: List[str],
        target_folder: str = settings.DEFAULT_DATA_DIRECTORY,
    ) -> List[str]:
        """
        Download files or directories from S3 to a specific sandbox.
        Automatically detects whether each S3 path is a file or directory.

        Args:
            sandbox_id: Unique identifier for the sandbox
            s3_paths: List of S3 paths (keys or prefixes) to download
            target_folder: Target folder in sandbox
        Returns:
            List of downloaded file paths
        """
        self._validate_sandbox_exists(sandbox_id)

        if not settings.FILE_STORAGE_ENABLED:
            logger.error("File storage is not enabled. Cannot download from S3.")
            raise ValueError("File storage is not enabled. Cannot download from S3.")

        if (
            not settings.S3_BUCKET
            or not settings.S3_ACCESS_KEY_ID
            or not settings.S3_SECRET_ACCESS_KEY
        ):
            logger.error("S3 configuration is incomplete. Cannot download from S3.")
            raise ValueError("S3 configuration is incomplete. Cannot download from S3.")

        logger.info(
            f"Downloading {len(s3_paths)} path(s) from S3 to sandbox {sandbox_id}"
        )

        env_vars = {
            "AWS_ACCESS_KEY_ID": settings.S3_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": settings.S3_SECRET_ACCESS_KEY,
        }

        downloaded_files = []
        for s3_path in s3_paths:
            filename = Path(s3_path).name
            target_path = f"{target_folder}/{filename}"

            cmd_args = [
                settings.S3_BUCKET,
                s3_path,
                target_path,
            ]

            if settings.S3_ENDPOINT:
                cmd_args.extend(["--endpoint", settings.S3_ENDPOINT])

            result = self._run_script_in_sandbox(
                sandbox_id,
                "app/scripts/s3_download.py",
                "/tmp/s3_download.py",
                cmd_args,
                "sudo -E python",
                env_vars,
            )

            if result.error:
                logger.error(f"Error downloading {s3_path}: {result.error}")
                raise RuntimeError(f"Failed to download {s3_path}: {result.error}")

            logger.info(f"Download result: {result.stdout}")

            downloaded_files.append(target_path)
            logger.info(f"Downloaded {s3_path} to {target_path}")

        logger.info(f"Successfully downloaded {len(downloaded_files)} path(s)")
        return downloaded_files

    def upload_to_s3(
        self,
        sandbox_id: str,
        source_path: str,
        s3_path: str,
        delete_source: bool = True,
    ) -> None:
        """
        Upload a file or directory from sandbox to S3.

        Args:
            sandbox_id: Unique identifier for the sandbox
            source_path: Path of the file/folder to upload in sandbox
            s3_path: Target S3 path (key or prefix)
            delete_source: Whether to delete source after upload
        """
        self._validate_sandbox_exists(sandbox_id)
        sandbox = self.sandboxes[sandbox_id]

        if not settings.FILE_STORAGE_ENABLED:
            logger.error("File storage is not enabled. Cannot upload to S3.")
            raise ValueError("File storage is not enabled. Cannot upload to S3.")

        if (
            not settings.S3_BUCKET
            or not settings.S3_ACCESS_KEY_ID
            or not settings.S3_SECRET_ACCESS_KEY
        ):
            logger.error("S3 configuration is incomplete. Cannot upload to S3.")
            raise ValueError("S3 configuration is incomplete. Cannot upload to S3.")

        if not sandbox.files.exists(source_path):
            logger.error(
                f"Source path '{source_path}' does not exist in sandbox {sandbox_id}"
            )
            raise FileNotFoundError(f"Source path '{source_path}' does not exist")

        logger.info(
            f"Uploading {source_path} to S3 path {s3_path} in sandbox {sandbox_id}"
        )

        cmd_args = [
            source_path,
            settings.S3_BUCKET,
            s3_path,
        ]

        if settings.S3_ENDPOINT:
            cmd_args.extend(["--endpoint", settings.S3_ENDPOINT])

        env_vars = {
            "AWS_ACCESS_KEY_ID": settings.S3_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": settings.S3_SECRET_ACCESS_KEY,
        }

        result = self._run_script_in_sandbox(
            sandbox_id,
            "app/scripts/s3_upload.py",
            "/tmp/s3_upload.py",
            cmd_args,
            "sudo -E python",
            env_vars,
        )

        if result.error:
            logger.error(f"Error uploading {source_path}: {result.error}")
            raise RuntimeError(f"Failed to upload {source_path}: {result.error}")

        logger.info(
            f"Successfully uploaded {source_path} to s3://{settings.S3_BUCKET}/{s3_path}"
        )

        if delete_source:
            sandbox.files.remove(source_path)
            logger.info(f"Deleted source file: {source_path}")

    def print_limited_tree(
        self, sandbox_id: str, directory: str = settings.DEFAULT_WORKING_DIRECTORY
    ) -> str:
        """
        Print a limited tree of files in a specific sandbox directory.

        Args:
            sandbox_id: Unique identifier for the sandbox
            directory: Directory to list files from
        Returns:
            String representation of the limited tree
        """
        self._validate_sandbox_exists(sandbox_id)

        logger.info(f"Listing files in directory {directory} of sandbox {sandbox_id}")

        result = self._run_script_in_sandbox(
            sandbox_id,
            "app/scripts/limited_tree.sh",
            "/tmp/limited_tree.sh",
            [directory],
            "sudo",
            None,
        )

        if result.error:
            logger.error(f"Error listing files in sandbox {sandbox_id}: {result.error}")
            return ""

        logger.info(f"Files listed successfully in sandbox {sandbox_id}")
        return result.stdout

    def path_exists(self, sandbox_id: str, path: str) -> bool:
        """
        Check if a path exists in a specific sandbox.

        Args:
            sandbox_id: Unique identifier for the sandbox
        Returns:
            True if the path exists, False otherwise
        """
        self._validate_sandbox_exists(sandbox_id)

        sandbox = self.sandboxes[sandbox_id]

        exists = sandbox.files.exists(path)
        logger.info(
            f"Path existence check for '{path}' in sandbox '{sandbox_id}': {exists}"
        )
        return exists

    def destroy_all_sandboxes(self):
        """Destroy all sandboxes."""
        logger.info(f"Destroying all sandboxes ({len(self.sandboxes)} total)")
        for sandbox in self.sandboxes.values():
            sandbox.kill()
        self.sandboxes.clear()
        self.contexts.clear()
        logger.info("All sandboxes destroyed")
