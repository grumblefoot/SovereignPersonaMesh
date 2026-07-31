"""
Rotating file logger for SPM Proxy (Port 5050).

Outputs to sys.stdout and a rotating log file at logs/spm_proxy.log
(10 MB max per file, 5 backup files).
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Ensure the logs directory exists
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_LOG_FILE = os.path.join(_LOG_DIR, "spm_proxy.log")
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_spm_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root SPM logger with stdout + rotating file handlers."""
    logger = logging.getLogger("SPM")
    logger.setLevel(level)

    # Avoid duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT)

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
