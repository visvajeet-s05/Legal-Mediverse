"""
Production Logging Configuration
=================================
Structured JSON logging for cloud-native log aggregation (CloudWatch, Loki, Datadog).
"""

import logging
import sys
from pythonjsonlogger import jsonlogger


def setup_production_logging(name: str = "escrow_indexer") -> logging.Logger:
    """
    Configure and return a structured JSON logger suitable for production.
    All log entries will be emitted as JSON objects for easy querying in
    log management systems.

    Args:
        name: Logger name, defaults to 'escrow_indexer'.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove any pre-existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(filename)s %(lineno)d",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent log propagation to root logger
    logger.propagate = False

    return logger


# Pre-configured module-level loggers for direct import
escrow_indexer_logger = setup_production_logging("escrow_indexer")
api_logger = setup_production_logging("api")

