"""Tiny logging helper and error counter.

Kept intentionally simple:
- Console logger only (easy to read)
- ErrorTracker is an in-memory counter for quick monitoring
"""

import logging
import os
from collections import defaultdict

LOGS_DIR = os.getenv("LOGS_DIR", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a simple console logger.

    Use this in router/worker to keep logs readable during development.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(console)

    logger.propagate = False
    return logger


class ErrorTracker:
    """Tiny in-memory error counter."""

    def __init__(self) -> None:
        self._errors = defaultdict(int)

    def record(self, key: str) -> None:
        self._errors[key] += 1

    def get_summary(self) -> dict:
        return dict(self._errors)

    def clear(self) -> None:
        self._errors.clear()


error_tracker = ErrorTracker()
