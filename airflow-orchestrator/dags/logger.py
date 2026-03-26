import logging
import os
import sys

from pythonjsonlogger.json import JsonFormatter


class Logger:
    def __new__(cls, *args, **kwargs) -> "Logger":
        if not hasattr(cls, "_instance"):
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return None

        # Get log level from env
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            log_level = "INFO"

        # Initialize the main logger with a default name
        self.logger = logging.getLogger("cognibrew")
        self.logger.setLevel(getattr(logging, log_level))

        if not self.logger.handlers:
            formatter = JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={
                    "asctime": "timestamp",
                    "levelname": "level",
                    "name": "logger",
                },
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self._initialized = True

    def get_logger(self) -> logging.Logger:
        """Return a logger instance named after the calling file."""
        frame = sys._getframe(1)
        filepath = frame.f_globals.get("__file__", "cognibrew")
        name = os.path.basename(filepath)
        file_logger = logging.getLogger(name)

        if not file_logger.handlers:
            file_logger.setLevel(self.logger.level)
            file_logger.propagate = False
            for handler in self.logger.handlers:
                file_logger.addHandler(handler)

        return file_logger
