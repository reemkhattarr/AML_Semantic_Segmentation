import logging
import os
from typing import Optional

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S"
) -> logging.Logger:
    """
    Set up and return a logger with the specified name.
    Logs to both console and (optionally) a file.
    Prevents duplicate handlers if called multiple times.

    Args:
        name (str): Name of the logger.
        log_file (Optional[str]): Path to a log file. If None, file logging is disabled.
        level (int): Logging level (e.g., logging.INFO).
        fmt (str): Log message format.
        datefmt (str): Date format for log messages.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Prevent adding multiple handlers in interactive environments
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler (optional)
        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, mode='a')
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    # Avoid propagating to root logger
    logger.propagate = False
    return logger
