"""Unified logging configuration for router-maestro."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from router_maestro.config.paths import get_data_dir


def get_log_dir() -> Path:
    """Get the log directory ~/.local/share/router-maestro/logs/."""
    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
) -> None:
    """Configure unified logging with console (Rich) and file output.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        console: Enable Rich console handler with colors
        file: Enable rotating file handler
    """
    # Get the root logger for router_maestro
    logger = logging.getLogger("router_maestro")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on reconfig
    logger.handlers.clear()

    # File format (no colors)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (Rich with colors)
    if console:
        console_handler = RichHandler(
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(console_handler)

    # File handler (rotating, 10MB max, 5 backups)
    if file:
        log_file = get_log_dir() / "router-maestro.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (will be prefixed with 'router_maestro.')

    Returns:
        Logger instance for the module
    """
    return logging.getLogger(f"router_maestro.{name}")
