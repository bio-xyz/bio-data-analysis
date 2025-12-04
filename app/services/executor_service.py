import os
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

    def mount_s3_bucket(
        self,
        sandbox_id: str,
        mount_point: str = settings.DEFAULT_MOUNT_DIRECTORY,
    ) -> str:
        """
        Mount S3 bucket in a specific sandbox.
        Args:
            sandbox_id: Unique identifier for the sandbox
            mount_point: Mount point in the sandbox
        """
        self._validate_sandbox_exists(sandbox_id)

        sandbox = self.sandboxes[sandbox_id]

        if (
            settings.FILE_STORAGE_ENABLED is False
            or not settings.S3_BUCKET
            or not settings.S3_ACCESS_KEY_ID
            or not settings.S3_SECRET_ACCESS_KEY
        ):
            logger.error("S3 configuration is incomplete. Cannot mount S3 bucket.")
            raise ValueError("S3 configuration is incomplete. Cannot mount S3 bucket.")

        logger.info(f"Mounting S3 bucket in sandbox {sandbox_id} at {mount_point}")
        sandbox.files.write(
            "/root/.passwd-s3fs",
            f"{settings.S3_ACCESS_KEY_ID}:{settings.S3_SECRET_ACCESS_KEY}",
        )
        sandbox.commands.run("sudo chmod 600 /root/.passwd-s3fs")

        sandbox.files.make_dir(mount_point)
        sandbox.commands.run(
            f"sudo s3fs {settings.S3_BUCKET} {mount_point} "
            "-o passwd_file=/root/.passwd-s3fs "
            "-o use_path_request_style "
            f"{f'-o url={settings.S3_ENDPOINT} ' if settings.S3_ENDPOINT else ''}"
            "-o allow_other "
            "-o sync"
        )

        logger.info(f"S3 bucket mounted successfully in sandbox {sandbox_id}")
        return mount_point

    def unmount_s3_bucket(
        self,
        sandbox_id: str,
        mount_point: str = settings.DEFAULT_MOUNT_DIRECTORY,
    ):
        """
        Unmount S3 bucket in a specific sandbox.
        Args:
            sandbox_id: Unique identifier for the sandbox
            mount_point: Mount point in the sandbox
        """
        self._validate_sandbox_exists(sandbox_id)

        sandbox = self.sandboxes[sandbox_id]

        logger.info(f"Unmounting S3 bucket in sandbox {sandbox_id} from {mount_point}")

        sandbox.commands.run(f"sudo sync")
        sandbox.commands.run(f"sudo umount {mount_point}")

        sandbox.files.remove("/root/.passwd-s3fs")

        logger.info(f"S3 bucket unmounted successfully in sandbox {sandbox_id}")

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

    def copy_from_mount(
        self,
        sandbox_id: str,
        file_paths: List[str],
        target_folder: str = settings.DEFAULT_DATA_DIRECTORY,
        mount_folder: str = settings.DEFAULT_MOUNT_DIRECTORY,
    ) -> List[str]:
        """
        Download and copy file paths to a specific sandbox's user_data folder from s3 mount.

        Args:
            sandbox_id: Unique identifier for the sandbox
            file_paths: List of files to be copiled from mount
            target_folder: Target folder in sandbox
        Returns:
            List of copied file paths
        """
        self._validate_sandbox_exists(sandbox_id)
        sandbox = self.sandboxes[sandbox_id]

        if settings.FILE_STORAGE_ENABLED is False:
            logger.error("File storage is not enabled. Cannot copy from mount.")
            raise ValueError("File storage is not enabled. Cannot copy from mount.")

        logger.info(f"Copying {len(file_paths)} file to sandbox {sandbox_id}")

        sandbox.files.make_dir(target_folder)

        registered_files = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            target_path = f"{target_folder}/{filename}"

            sandbox.commands.run(f"sudo cp -r {mount_folder}/{file_path} {target_path}")
            registered_files.append(target_path)
            logger.info(f"Copied file path: {file_path} to {target_path}")

        logger.info(f"Successfully copied {len(registered_files)} files")
        return registered_files

    def move_to_mount(
        self,
        sandbox_id: str,
        source_path: Path,
        target_path: Path,
        mount_folder: Path = Path(settings.DEFAULT_MOUNT_DIRECTORY),
    ):
        """
        Copy files to a specific sandbox's s3 mount folder.

        Args:
            sandbox_id: Unique identifier for the sandbox
            source_path: Path of the file/folder to move
            target_path: Target path in mount
            mount_folder: Mount folder in sandbox
        Returns:
            List of uploaded file paths
        """
        self._validate_sandbox_exists(sandbox_id)
        sandbox = self.sandboxes[sandbox_id]

        if not settings.FILE_STORAGE_ENABLED:
            logger.error("File storage is not enabled. Cannot copy to mount.")
            raise ValueError("File storage is not enabled. Cannot copy to mount.")

        logger.info(f"Uploading file to mount in sandbox {sandbox_id}")

        target_path = mount_folder / target_path
        sandbox.files.make_dir(target_path.parent.as_posix())

        copy_result = sandbox.commands.run(f"sudo cp -r {source_path} {target_path}")
        if copy_result.error:
            logger.error(
                f"Error moving file to mount in sandbox {sandbox_id}: {copy_result.error}"
            )
            return

        sandbox.files.remove(source_path.as_posix())

        logger.info(f"Uploaded file path: {source_path} to {target_path}")

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

        sandbox = self.sandboxes[sandbox_id]

        logger.info(f"Listing files in directory {directory} of sandbox {sandbox_id}")

        with open("app/scripts/limited_tree.sh", "r") as script_file:
            script_content = script_file.read()

        script_path = "/tmp/limited_tree.sh"
        sandbox.files.write(script_path, script_content)
        sandbox.commands.run(f"sudo chmod +x {script_path}")

        result = sandbox.commands.run(f"{script_path} {directory}")

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
