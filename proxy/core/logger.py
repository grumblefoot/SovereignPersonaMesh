"""
Rotating Log Handler for Sovereign Persona Mesh (SPM).

Configures a RotatingFileHandler that writes structured logs to
logs/spm_proxy.log with a 10 MB rotation limit.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
LOG_FILE = os.path.join(LOG_DIR, "spm_proxy.log")
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_rotating_logger(
    name: str = "SPMProxy",
    level: int = logging.INFO,
    log_file: str = LOG_FILE,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> logging.Logger:
    """
    Configure and return a logger that writes to a rotating log file.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level (default: INFO).
    log_file : str
        Path to the log file.
    max_bytes : int
        Max bytes per log file before rotation.
    backup_count : int
        Number of rotated backup files to keep.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    # Also keep a console handler for debugging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    logger.info(f"[Logger] Rotating log handler configured -> {log_file}")
    return logger


setup_spm_logging = setup_rotating_logger
