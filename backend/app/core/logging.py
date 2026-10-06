"""
SolarSentinel AI - Structured Logging Setup
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures structured console logging."""
    logger = logging.getLogger("solarsentinel")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s (%(module)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
