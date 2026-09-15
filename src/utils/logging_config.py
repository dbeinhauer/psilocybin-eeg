"""
Logging configuration for EEG analysis project.

This module provides centralized logging configuration with different
handlers for console, file, and error logging.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    module_name: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration for the project.

    :param log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
    :param log_dir: Directory for log files. If None, uses 'logs/' in current directory
    :param log_to_file: Whether to log to file
    :param log_to_console: Whether to log to console
    :param module_name: Name of the module logger. If None, configures root logger
    :return: Configured logger instance
    """
    # Get logger
    logger = logging.getLogger(module_name) if module_name else logging.getLogger()
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File handlers
    if log_to_file:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file (all messages)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_log_file = log_dir / f"eeg_analysis_{timestamp}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,  # 10 MB
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        # Error log file (errors and critical only)
        error_log_file = log_dir / f"eeg_analysis_errors_{timestamp}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,  # 10 MB
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    :param name: Name of the module (typically __name__)
    :return: Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.

    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Doing something")
    """

    @property
    def logger(self) -> logging.Logger:
        """
        Get logger for this class.

        :return: Logger instance
        """
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return logging.getLogger(name)


# Logging context manager for temporary log level changes
class LogLevel:
    """
    Context manager for temporarily changing log level.

    Usage:
        with LogLevel(logging.DEBUG):
            # Code with DEBUG level logging
            logger.debug("Debug message")
        # Back to original level
    """

    def __init__(self, level: int, logger: Optional[logging.Logger] = None):
        """
        Initialize context manager.

        :param level: Temporary log level
        :param logger: Logger to modify. If None, uses root logger
        """
        self.level = level
        self.logger = logger or logging.getLogger()
        self.original_level = None

    def __enter__(self):
        """Enter context - save current level and set new level."""
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original level."""
        self.logger.setLevel(self.original_level)


def setup_testing_logging() -> logging.Logger:
    """
    Set up logging for testing (warnings and errors only).

    :return: Configured logger
    """
    return setup_logging(
        log_level=logging.WARNING, log_to_file=False, log_to_console=True
    )
