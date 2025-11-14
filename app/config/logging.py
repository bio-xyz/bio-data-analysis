import logging

from app.config.settings import settings


def setup_logging():
    """Configure logging with colored output."""
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Add color to log levels
    logging.addLevelName(logging.DEBUG, "\033[36mDEBUG\033[0m")
    logging.addLevelName(logging.INFO, "\033[32mINFO\033[0m")
    logging.addLevelName(logging.WARNING, "\033[33mWARNING\033[0m")
    logging.addLevelName(logging.ERROR, "\033[31mERROR\033[0m")
    logging.addLevelName(logging.CRITICAL, "\033[35mCRITICAL\033[0m")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
