"""Data file utilities for handling file uploads and validation."""

from typing import Optional

from fastapi import HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.config import get_logger, settings

logger = get_logger(__name__)


class DataFile(BaseModel):
    """
    Model representing a data file with its content, metadata, and validation.

    Attributes:
        filename: Name of the file
        content: Raw bytes content of the file
        size: Size of the file in bytes
        content_type: MIME type of the file (optional)
    """

    filename: str = Field(..., description="Name of the file")
    content: bytes = Field(..., description="Raw bytes content of the file")
    size: int = Field(..., description="Size of the file in bytes", ge=0)
    content_type: Optional[str] = Field(
        None, description="MIME type of the file (optional)"
    )


def validate_file_size(size: int, max_size: int) -> None:
    """
    Validate that file size is within allowed limits.

    Args:
        size: File size in bytes
        max_size: Maximum allowed file size in bytes

    Raises:
        HTTPException: If file size exceeds the maximum allowed size
    """
    if size > max_size:
        size_mb = size / (1024 * 1024)
        max_size_mb = max_size / (1024 * 1024)
        error_msg = f"File size {size_mb:.2f}MB exceeds maximum allowed size of {max_size_mb:.2f}MB"
        logger.warning(error_msg)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=error_msg,
        )


async def convert_upload_file_to_data_file(
    upload_file: UploadFile,
    max_size: Optional[int] = None,
) -> DataFile:
    """
    Convert FastAPI UploadFile to DataFile with validation.

    Args:
        upload_file: FastAPI UploadFile object
        max_size: Maximum allowed file size in bytes (defaults to settings.MAX_FILE_SIZE)
        allowed_extensions: List of allowed extensions (defaults to settings.ALLOWED_FILE_EXTENSIONS)

    Returns:
        DataFile: Validated DataFile object

    Raises:
        HTTPException: If file validation fails
    """
    if max_size is None:
        max_size = settings.MAX_FILE_SIZE

    filename = upload_file.filename or "unnamed_file"

    # Validate file size
    size = upload_file.size
    validate_file_size(size, max_size)

    # Read file content
    content = await upload_file.read()

    logger.info(f"Converted file '{filename}' to DataFile (size: {size} bytes)")

    return DataFile(
        filename=filename,
        content=content,
        size=size,
        content_type=upload_file.content_type,
    )


async def convert_upload_files_to_data_files(
    upload_files: list[UploadFile],
    max_size: Optional[int] = None,
    allowed_extensions: Optional[list[str]] = None,
) -> list[DataFile]:
    """
    Convert multiple FastAPI UploadFiles to DataFiles with validation.

    Args:
        upload_files: List of FastAPI UploadFile objects
        max_size: Maximum allowed file size in bytes (defaults to settings.MAX_FILE_SIZE_MB)

    Returns:
        List of validated DataFile objects

    Raises:
        HTTPException: If any file validation fails
    """
    data_files = []
    for upload_file in upload_files:
        data_file = await convert_upload_file_to_data_file(upload_file, max_size)
        data_files.append(data_file)

    logger.info(f"Converted {len(data_files)} files to DataFiles")
    return data_files
