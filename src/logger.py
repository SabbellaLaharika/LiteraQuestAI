import sys
from loguru import logger
from src.config import LOG_FILE

# Remove any default loguru handlers
logger.remove()

# Structured log format: timestamp | level | message
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

FILE_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

# Console handler (standard output)
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level="INFO",
    colorize=True,
    enqueue=True,
)

# Mandatory file handler writing to ./output/agent_run.log
logger.add(
    str(LOG_FILE),
    format=FILE_LOG_FORMAT,
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    enqueue=True,
    encoding="utf-8",
)

__all__ = ["logger"]
