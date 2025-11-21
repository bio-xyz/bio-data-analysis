"""Security utilities for API authentication."""

import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import get_logger

logger = get_logger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key_from_env() -> Optional[str]:
    """
    Get the API key from environment variables.

    Returns:
        Optional[str]: The API key if configured, None otherwise.
    """
    return os.getenv("API_KEY")


async def validate_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Validate the API key from request headers.

    This dependency can be used on any endpoint that requires API key authentication.
    Add it as a dependency using FastAPI's Depends() or Security().

    Args:
        api_key: The API key from the X-API-Key header.

    Returns:
        str: The validated API key.

    Raises:
        HTTPException: 401 if API key is missing or invalid, 500 if not configured.
    """
    expected_api_key = get_api_key_from_env()

    if not expected_api_key:
        logger.warning("API_KEY environment variable is not configured")
        return ""

    if not api_key:
        logger.debug("API key missing in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

    if api_key != expected_api_key:
        logger.debug("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )

    logger.debug("API key validated successfully")
    return api_key
