"""Zen logging - minimal, clean, effective."""

import logging
import logging.handlers
import os
from collections import defaultdict

# Create logs directory if it doesn't exist
os.makedirs(os.getenv("LOGS_DIR", "logs"), exist_ok=True)


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a simple logger with console and file output.

    Args:
        name: Logger name (usually module or component name).
        level: Logging level (DEBUG/INFO/WARNING/ERROR).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    # Skip if already configured (prevents duplicate handlers)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console output: human-readable format with timestamp
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)

    # File output: rotating logs to prevent disk space issues
    # Max 10MB per file, keep 3 backups
    log_file = os.path.join(os.getenv("LOGS_DIR", "logs"), f"{name}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return logger


class ErrorTracker:
    """
    Simple error counter for tracking error frequency.

    Provides insight into system reliability without heavy instrumentation.
    """

    def __init__(self) -> None:
        """Initialize error tracker with empty counter."""
        # defaultdict(int) allows missing keys to default to 0
        self.errors: dict[str, int] = defaultdict(int)

    def record(self, error_type: str) -> None:
        """Record an error occurrence.

        Args:
            error_type: Type of error to record.
        """
        self.errors[error_type] += 1

    def get_summary(self) -> dict:
        """Get error summary.

        Returns:
            Dictionary mapping error_type to count.
        """
        return dict(self.errors)

    def clear(self) -> None:
        """Clear all error counters."""
        self.errors.clear()


# Global error tracker instance used by router and workers
error_tracker = ErrorTracker()
