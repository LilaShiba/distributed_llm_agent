"""Zen logging - minimal, clean, effective."""

import logging
import logging.handlers
import os
from collections import defaultdict

os.makedirs(os.getenv("LOGS_DIR", "logs"), exist_ok=True)


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup a simple logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    
    # File
    log_file = os.path.join(os.getenv("LOGS_DIR", "logs"), f"{name}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=3
    )
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)
    
    return logger


class ErrorTracker:
    """Simple error counter."""
    
    def __init__(self):
        self.errors = defaultdict(int)
    
    def record(self, error_type: str):
        """Record an error."""
        self.errors[error_type] += 1
    
    def get_summary(self) -> dict:
        """Get error summary."""
        return dict(self.errors)
    
    def clear(self):
        """Clear counters."""
        self.errors.clear()


error_tracker = ErrorTracker()
