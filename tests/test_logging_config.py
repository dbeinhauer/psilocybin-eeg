"""
Tests for src/utils/logging_config.py — Logging setup and helpers.
"""

import pytest
import logging
from pathlib import Path

from src.utils.logging_config import (
    setup_logging,
    get_logger,
    LoggerMixin,
    LogLevel,
    setup_testing_logging,
)


class TestSetupLogging:
    """Test setup_logging function."""

    def test_returns_logger(self):
        logger = setup_logging(log_to_file=False, log_to_console=True)
        assert isinstance(logger, logging.Logger)

    def test_console_handler_added(self):
        logger = setup_logging(
            log_to_file=False, log_to_console=True, module_name="test_console"
        )
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types

    def test_file_handler_created(self, tmp_path):
        logger = setup_logging(
            log_to_file=True,
            log_to_console=False,
            log_dir=tmp_path,
            module_name="test_file",
        )
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "RotatingFileHandler" in handler_types
        # Should have 2 file handlers (main + error)
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 2

    def test_log_level_set(self):
        logger = setup_logging(
            log_level=logging.DEBUG,
            log_to_file=False,
            log_to_console=True,
            module_name="test_level",
        )
        assert logger.level == logging.DEBUG


class TestGetLogger:
    def test_returns_named_logger(self):
        logger = get_logger("my_module")
        assert logger.name == "my_module"

    def test_returns_logger_instance(self):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)


class TestLoggerMixin:
    def test_mixin_provides_logger(self):
        class MyClass(LoggerMixin):
            pass

        obj = MyClass()
        assert isinstance(obj.logger, logging.Logger)
        assert "MyClass" in obj.logger.name


class TestLogLevel:
    def test_context_manager_changes_level(self):
        logger = logging.getLogger("test_ctx")
        logger.setLevel(logging.INFO)
        with LogLevel(logging.DEBUG, logger):
            assert logger.level == logging.DEBUG
        assert logger.level == logging.INFO

    def test_context_manager_restores_on_exception(self):
        logger = logging.getLogger("test_ctx_exc")
        logger.setLevel(logging.WARNING)
        try:
            with LogLevel(logging.DEBUG, logger):
                raise ValueError("test")
        except ValueError:
            pass
        assert logger.level == logging.WARNING


class TestSetupTestingLogging:
    def test_warning_level(self):
        logger = setup_testing_logging()
        assert logger.level == logging.WARNING
