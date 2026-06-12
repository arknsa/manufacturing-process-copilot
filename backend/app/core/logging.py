"""
backend/app/core/logging.py
============================
Structured logging configuration. Call configure_logging() once
at application startup (inside lifespan).
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Suppress noisy third-party loggers
    for noisy in ("mlflow", "urllib3", "botocore", "git", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
