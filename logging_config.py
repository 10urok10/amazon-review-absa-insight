"""
Shared logging setup. Every module gets a logger writing to both the console
and app.log with a consistent format, instead of ad-hoc print() calls for
diagnostics (warnings, retries, errors). User-facing pipeline progress
narration still goes through the `progress` callback pattern in
insight_engine.py (which needs to reach different UIs -- CLI stdout, a
Streamlit placeholder, etc.) -- this is for the operational record.
"""

import logging

from config import LOG_PATH

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured -- avoid duplicate handlers on re-import

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
